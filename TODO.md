

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
- compile

TRACK:
- mfu
- im/s
- memory
- time
