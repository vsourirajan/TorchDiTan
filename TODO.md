

- [] Set up data loading 
- [] Patchify/Depatchify
- [] L2 Loss computation



Alper's notes on dataloading:

- Assume PP and CP are disabled!! These makes things too complicated
- Surprisingly, all training happens in train.py
- The data is loaded as inputs,labels from the dataloader. These tensors appear to be in the CPU until they go into the model.
- Created `diffusion_model.py` in models/llama
- The new entry point to the code is `./run_llama_train_diffusion`.

Alper backburner:
- Maybe properly initialize weights of the patch emebdder?

Alper todo:
- make new dataloader to produce (partially noisy image x_t scalar t), x1-x0 direction

data_entries
{
    "input":Tensor[B, C, H, W], (float)
    "class": Tensor[B, ] (integer)
    "time" Tensor[B, ] (float)
}
- modify training loop

Vaibhav todo:
- change model architecture to accept partially noisy image, scalar t  --> direction (same shape as noisy image)
    - dimensions (bs, num_patches + t_dim, embed_dim) --> (bs, num_patches + t, embed_dim) --> (cut off ts before depatchify)
    - implement sinusoidal positional encoding to project float t into embed_dim and concat
    - condition on class by concatting learnable vector that projects class to embed_dim

Shreya todo:
- make slides
- remove upper triangular attention mask



ABLATE:
- dtype
- replicate/shard degree
- activation checkpointing
- model size (various sizes, vary depth vs width while keeping size constant)
- patch size
- batch size (?)
- float 8 linear
- compile
- compiled autograd
- compiled optimizer

TRACK:
- mfu
- im/s
- memory
- time
- loss for 1000 steps







You think training has slowed down? Check this :

IT'S IMPORTANT THAT YOU TURN OFF CAUSAL ATTENTION, WITH CAUSAL ATTENTION, THE ATTENTION LAYERS ARE WAY FASTER

CUDA_VISIBLE_DEVICES=6,7 NGPU=2 ./run_llama_train_diffusion.sh

"llama3_diffusion_small": DiffusionModelArgs(dim=256, n_layers=16, n_heads=16, rope_theta=500000, patch_size=2),

[rank0]:2024-12-13 17:29:23,398 - root - INFO - step: 30  loss:  1.7148  memory: 14.91GiB(18.82%)  wps: 48,655  mfu: 13.87% it/s: 0.37 im/s: 5.94

[dataset]
batch_size = -1 # get it from training.batch_size
dataset_name = "imagenet"
root_dir = "/local/vondrick/datasets/imagenet"
num_workers = 4
image_size = [256, 256]
num_classes = 1000

[model]
name = "llama3_diffusion"
flavor = "llama3_diffusion_small"

[training]
batch_size = 8
seq_len = 16386

[rank0]:doing attention with h shape:  torch.Size([16, 16386, 256])