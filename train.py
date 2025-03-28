# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import time
from datetime import timedelta

import torch
import wandb

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)

from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.samplers import sample_and_visualize

from data import build_image_dataloader
from dataclasses import asdict
from cosmos_latent_decoder import CosmosDecoder

def get_config_dict(config: JobConfig) -> dict:
    """Convert JobConfig object to a flat dictionary for wandb logging"""
    config_dict = {}
    
    print("at the beginning, does config have job?", hasattr(config, 'job'))
    # Iterate through all sections of the config
    for section_name in ['job', 'model', 'training', 'optimizer', 'metrics', 
                        'profiling', 'checkpoint', 'experimental', 'float8']:
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            # Add all attributes from the section to the flat dict
            for key, value in section.__dict__.items():
                config_dict[f"{section_name}.{key}"] = value

    return config_dict

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig, args_dict: dict): #args dict is a hack, it's job config in dict form
    init_logger()
    
    # Initialize wandb before distributed init
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Initialize wandb with unique name for each process
    config_dict = get_config_dict(job_config)
    if rank == 0:  # Only initialize wandb on the main process
        wandb.init(
            project="torchditan_latent",
            config=config_dict,
            group=job_config.job.description,  
            tags=[f"world_size_{world_size}", f"{job_config.dataset.dataset_name}"],
            mode='online' if job_config.metrics.enable_wandb else 'disabled'
        )

    logger.info(f"Starting job: {job_config.job.description}")
    
    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # set determinisism, use seed == None to skip deterministic training
    utils.set_determinism(job_config.training.seed)
    if job_config.training.seed is None:
        logger.info("Deterministic training off")
    else:
        logger.info(
            f"Deterministic training on. Using seed: {job_config.training.seed}"
        )

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    print("DP SHARD: ", parallel_dims.dp_shard)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    if job_config.dataset.batch_size == -1:
        job_config.dataset.batch_size = job_config.training.batch_size
    data_loader, sampler, classes = build_image_dataloader(job_config.dataset)

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

   
    #compute max seq len
    fixed_timestep_and_label_embedder_size = 2
    patch_size = job_config.model.patch_size
    image_size = job_config.dataset.image_size
    num_classes = job_config.dataset.num_classes
    max_seq_len = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    
    if model_config.condition_mode == "context":
        max_seq_len += fixed_timestep_and_label_embedder_size
    
    model_config.max_seq_len = max_seq_len
    job_config.training.seq_len = max_seq_len

    model_config.image_size = image_size
    model_config.num_classes = num_classes
    model_config.input_channels = job_config.model.input_channels
    model_config.patch_size = patch_size
    latent_diffusion_enabled = job_config.model.input_channels > 3 #fair assumption
    print("[INFO] max_seq_len not specified, manually calculated to be", max_seq_len)

    latent_decoder = CosmosDecoder( #for visualization purposes
        is_continuous=True,  # Since we're using continuous latents
        device=device,
    ) if (latent_diffusion_enabled and job_config.metrics.enable_sampling) else None


    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    if job_config.training.compile:
        loss_fn = torch.compile(loss_fn)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = "cuda"
    else:
        init_device = "cuda"
        buffer_device = None

    print("[LOG] DATA PARALLEL SHARD DEGREE", job_config.training.data_parallel_shard_degree, "DATA PARALLEL REPLICATE DEGREE", job_config.training.data_parallel_replicate_degree)
    if parallel_dims.dp_shard == 1:
        param_dtype = torch.bfloat16 if job_config.training.mixed_precision_param == "bfloat16" else torch.float32
        model.to(param_dtype)
        print("[LOG] manually applied dtype to model")

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        pp_schedule, model_parts = models_pipelining_fns[model_name](
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            m.init_weights(buffer_device=buffer_device)
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        model.init_weights(buffer_device=buffer_device)
        model.train()

        model_parts = [model]

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load()

    if parallel_dims.pp_enabled and not checkpoint_loaded:
        # TODO: fix this by allowing each rank to set their own seed
        logger.warning(
            "Pipeline Parallelism is being used without a seed checkpoint. "
            "All the substages will be initialized with random weights with same RNG state which can affect convergence."
        )

 
    metric_logger = build_metric_logger(job_config, parallel_dims)

    if rank == 0:
        import json
        #dump the model config dict to a json file
        with open(metric_logger.log_dir + "/config.json", "w") as f:
            json.dump(args_dict, f)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    if sampler is not None:
        sampler.set_epoch(train_state.step)
    data_iterator = iter(data_loader)


    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    nimages_since_last_log = 0
    ntokens_since_last_log = 0
    steps_since_last_log = 0

    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()


    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Reset the iterator when it runs out
                if sampler is not None:
                    sampler.set_epoch(train_state.step)
                data_iterator = iter(data_loader)
                batch = next(data_iterator)

   
            ntokens_since_last_log += job_config.training.seq_len * job_config.training.batch_size
            nimages_since_last_log += job_config.training.batch_size
            steps_since_last_log += 1
            data_loading_times.append(time.perf_counter() - data_load_start)

            batch["original_input"] = batch["original_input"].cuda()
            batch["class_idx"] = batch["class_idx"].cuda()

            param_dtype = torch.bfloat16 if job_config.training.mixed_precision_param == "bfloat16" else torch.float32
            #print("PARAM DTYPE", param_dtype)
            batch["original_input"] = batch["original_input"].to(dtype=param_dtype)
            #batch["param_dtype"] = param_dtype

            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[batch["original_input"], batch["class_idx"], model.freqs_cis],
                    cp_seq_dims=[1, 1, 0],
                    cp_no_restore_buffers={batch["original_input"], batch["class_idx"]},
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                # is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                # with train_context(optional_context_parallel_ctx):
                #     if pp_mesh.get_local_rank() == 0:
                #         pp_schedule.step(input_ids)
                #     elif is_last_stage:
                #         losses = []
                #         pp_schedule.step(target=labels, losses=losses)
                #     else:
                #         pp_schedule.step()

                # # accumulate losses across pipeline microbatches
                # loss = (
                #     torch.mean(torch.stack(losses))
                #     if is_last_stage
                #     else torch.Tensor([-1.0])
                # )
                raise NotImplementedError("Pipeline parallelism not implemented for DiT")
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):

                    x1 = batch["original_input"]
                    x0 = torch.randn_like(x1).to(x1.device)
                    
                    bs = x1.shape[0]
                    t = torch.rand(bs, device=x1.device, dtype=param_dtype)
                    batch["time"] = t

                    xt = x0 + t.view(-1, 1, 1, 1) * (x1 - x0)
                    xt = xt.to(dtype=param_dtype)
                    batch["input"] = xt
                    
                    pred = model(batch) #b, c, h, w

                    loss = torch.nn.functional.mse_loss(pred, x1 - x0)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), job_config.training.max_norm, foreach=True
                )

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            if (train_state.step - 1) % job_config.metrics.sample_freq == 0:
                # Generate and visualize samples
                if job_config.metrics.enable_sampling:
                    vis_results = sample_and_visualize(
                        model=model,
                        batch=batch,
                        param_dtype=param_dtype,
                        classes=classes,
                        latent_decoder=latent_decoder 
                    )
                
                    # Log to wandb
                    if rank == 0:
                        wandb.log(vis_results, step=train_state.step)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, dp_mesh),
                        utils.dist_max(max_loss, dp_mesh),
                    )
                    global_nimages_since_last_log = nimages_since_last_log * dp_degree
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss
                    global_nimages_since_last_log = nimages_since_last_log

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                
                time_delta = time.perf_counter() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                its = steps_since_last_log / time_delta
                images_per_sec = global_nimages_since_last_log / time_delta
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "im/s": images_per_sec,
                    "mfu(%)": mfu,
                    "it/s": its,
                    "im/s": images_per_sec,
                    "num_flop_per_token": num_flop_per_token,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}wps: {round(wps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                    f"{color.red} it/s: {its:.2f}{color.reset}"
                    f"{color.red} im/s: {images_per_sec:.2f}{color.reset}"
                )
                
                if rank == 0:
                    wandb.log({
                        "step": train_state.step,
                        "loss": global_avg_loss,
                        "memory_max_reserved_gib": gpu_mem_stats.max_reserved_gib,
                        "memory_max_reserved_pct": gpu_mem_stats.max_reserved_pct,
                        "wps": round(wps),
                        "mfu": mfu,
                        "its": its,
                        "im_s": images_per_sec,
                        "num_flop_per_token": num_flop_per_token
                    })

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()
                steps_since_last_log = 0
                nimages_since_last_log = 0

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    args_dict = config.parse_args()
    main(config, args_dict)
    
    # Cleanup wandb at the end
    # if rank == 0:
    #     wandb.finish()
    torch.distributed.destroy_process_group()
