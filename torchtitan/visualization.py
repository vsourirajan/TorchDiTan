import torch
from PIL import Image 
import matplotlib.pyplot as plt  

def rf_sample_euler(model: torch.nn.Module, 
                    N: int = 100, 
                    batch_size: int = 16, 
                    device: str = "cuda",
                    ):

    #create a dummy batch
    batch = {
        "input": torch.randn(batch_size, model.input_channels, model.input_image_size[0], model.input_image_size[1], device=device),
        "class_idx": torch.randint(0, model.vocab_size, (batch_size,), device=device),
    }

    dt = 1. / N
    for i in range(N):
        batch["time"] = torch.ones((batch_size,1), device=device) * i / N
        pred = model(batch)
        batch["input"] = batch["input"].detach().clone() + pred * dt

    #push to CPU and save as image
    out = batch["input"].cpu() #B, C, H, W
    out = out.clamp(0, 1)
    out = out.permute(0, 2, 3, 1) 
    out = (out * 255).to(torch.uint8) 
    
    fig, axs = plt.subplots(1, batch_size, figsize=(10, 10))
    for i in range(batch_size):
        axs[i].imshow(out[i].numpy())
        axs[i].axis("off")
    plt.savefig(f"rf_sample_euler_{N}.png")
    plt.close()
