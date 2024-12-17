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
from contextlib import contextmanager


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
    initial_gpu_mem = gpu_mem_stats.max_reserved_gib  # Save the initial memory usage
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

    # Replace training loop with batch size autotuning
    logger.info("Starting batch size autotuning...")
    
    # Binary search parameters
    max_batch_size = 1024
    min_batch_size = 1
    found_valid_batch_size = False
    max_retries = 1  # Number of attempts at each batch size before declaring it invalid
    optimal_batch_size = None

    def reset_states():
        # More aggressive memory clearing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Make sure all CUDA operations are completed
        
        # Clear model gradients
        # Clear model states
        for m in model_parts:
            m.zero_grad(set_to_none=True)
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Clear optimizer states
        optimizers.zero_grad()
        for opt in optimizers.optimizers:
            # Clear optimizer state dict
            opt.state.clear()
        
        if sampler is not None:
            sampler.set_epoch(train_state.step)
        checkpoint.reset()
        
        # Reset tracking variables
        losses_since_last_log.clear()
        data_loading_times.clear()
        global nimages_since_last_log, ntokens_since_last_log, steps_since_last_log
        nimages_since_last_log = 0
        ntokens_since_last_log = 0
        steps_since_last_log = 0

        time.sleep(1)

    def try_batch_size(batch_size, data_iterator):
        """Try a specific batch size and return True if it works"""
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        reset_states()
        
        try:
            logger.info(f"Attempting training with batch size: {batch_size}") #important, don't remove
            # Get initial retry count
            initial_retries = torch.cuda.memory_stats()["num_alloc_retries"] if "num_alloc_retries" in torch.cuda.memory_stats() else 0
          
            for _ in range(3):

                batch = next(data_iterator)
                batch["original_input"] = batch["original_input"].cuda()
                batch["class_idx"] = batch["class_idx"].cuda()

                param_dtype = torch.bfloat16 if job_config.training.mixed_precision_param == "bfloat16" else torch.float32
                #print("PARAM DTYPE", param_dtype)
                batch["original_input"] = batch["original_input"].to(dtype=param_dtype)

                optimizers.zero_grad()
                
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

                with train_context(optional_context_parallel_ctx):
                    x1 = batch["original_input"][:1]
                    x0 = torch.randn_like(x1).to(x1.device)[:1]

                    x0 = x0.repeat(batch_size, 1, 1, 1)
                    x1 = x1.repeat(batch_size, 1, 1, 1)

                    t = torch.rand(batch_size, device=x1.device, dtype=param_dtype)
                    batch["time"] = t

                    xt = x0 + t.view(-1, 1, 1, 1) * (x1 - x0)
                    xt = xt.to(dtype=param_dtype)
                    batch["input"] = xt

                    batch["class_idx"] = torch.randint(0, 1000, (batch_size,)).to(x1.device)

                    pred = model(batch)
                    loss = torch.nn.functional.mse_loss(pred, x1 - x0)
                    
                    del pred
                    loss.backward()
                    
                    # Actually step the optimizer to build up optimizer state
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

            final_retries = torch.cuda.memory_stats()["num_alloc_retries"] if "num_alloc_retries" in torch.cuda.memory_stats() else 0
            retries = final_retries - initial_retries
            if retries > 0:
                raise RuntimeError("Memory allocation retries exceeded")
            if gpu_memory_monitor.get_peak_stats().max_reserved_pct > 90:
                raise RuntimeError("Max memory usage exceeded")

            logger.info(f"Batch size {batch_size}: Success - Memory allocation retries: {retries}, GPU memory usage: {gpu_memory_monitor.get_peak_stats().max_reserved_pct:.2f}%")
            return True

        except (torch.cuda.OutOfMemoryError, Exception) as e:
            if rank == 0:
                final_retries = torch.cuda.memory_stats()["num_alloc_retries"] if "num_alloc_retries" in torch.cuda.memory_stats() else 0
                retries = final_retries - initial_retries
                current_mem = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.info(f"Batch size {batch_size}: Failed OOM - Current reserved memory: {gpu_memory_monitor.get_peak_stats().max_reserved_pct:.2f}%, Retries: {retries}")
                else:
                    logger.info(f"Batch size {batch_size}: Failed {str(e)[:50]}... - Current reserved memory: {gpu_memory_monitor.get_peak_stats().max_reserved_pct:.2f}%, Retries: {retries}")
            return False

    def round_down_to_multiple(value, multiple):
        """Round down to nearest multiple"""
        return (value // multiple) * multiple

    def estimate_max_batch_size(initial_mem_gb, peak_mem_gb, current_batch_size, gpu_total_mem_gb, batch_size_multiple, safety_factor=0.5):
        """
        Estimate maximum possible batch size based on memory usage patterns, always rounding down
        """
        batch_mem_usage = peak_mem_gb - initial_mem_gb
        mem_per_sample = safety_factor * batch_mem_usage / current_batch_size
        available_mem = gpu_total_mem_gb - initial_mem_gb
        theoretical_max = int((available_mem / mem_per_sample))
        theoretical_max = round_down_to_multiple(theoretical_max, batch_size_multiple)
        return max(batch_size_multiple, min(theoretical_max, 1024))

    # Binary search for optimal batch size
    multiple = job_config.autotune.batch_size_multiple
    left = multiple
    right = round_down_to_multiple(max_batch_size, multiple)

    # Adjust start_guess to be a multiple (rounding down)
    start_guess = job_config.training.batch_size if job_config.training.batch_size != -1 else (left + right) // 2
    start_guess = round_down_to_multiple(start_guess, multiple)

    if try_batch_size(start_guess, data_iterator):
        optimal_batch_size = start_guess
        
        # Estimate maximum based on successful run
        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        estimated_max = estimate_max_batch_size(
            initial_gpu_mem,
            gpu_mem_stats.max_reserved_gib,
            start_guess,
            80.0,  # hack: hardcoded total memory for now
            multiple
        )
        
        right = min(estimated_max, max_batch_size)
        left = start_guess + multiple
        
        if rank == 0:
            logger.info(f"First successful batch size: {start_guess}")
            logger.info(f"Estimated maximum batch size: {right}")
    else:
        right = start_guess - multiple

    # Binary search with multiples
    while left <= right:
        mid = round_down_to_multiple((left + right) // 2, multiple)
        mid = max(multiple, mid)
        
        if try_batch_size(mid, data_iterator):
            optimal_batch_size = mid
            left = mid + multiple
        else:
            right = mid - multiple

    if optimal_batch_size is None:
        logger.error("Could not find a valid batch size. Even minimum batch size causes OOM.")
        raise RuntimeError("No valid batch size found")

    # Set the optimal batch size and log it
    job_config.training.batch_size = optimal_batch_size
        
    logger.info(f"Binary search complete. Optimal batch size found: {optimal_batch_size}")

    metric_logger.close()
    logger.info("Batch size autotuning completed")


if __name__ == "__main__":
    config = JobConfig()
    args_dict = config.parse_args()
    main(config, args_dict)
    torch.distributed.destroy_process_group()
