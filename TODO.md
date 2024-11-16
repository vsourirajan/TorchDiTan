

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