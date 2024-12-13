import torch
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

@torch.no_grad()
def rf_sample_euler(model: torch.nn.Module, 
                    N: int = 100, 
                    batch_size: int = 16, 
                    device: str = "cuda",
                    classes: List[str] = None,
                    ):

    print("sampling...")
    # create a dummy batch
    class_indices = torch.randint(0, model.num_classes, (batch_size,), device=device)
    batch = {
        "input": torch.randn(batch_size, model.input_channels, model.input_image_size[0], model.input_image_size[1], device=device),
        "class_idx": class_indices,
    }

    dt = 1. / N
    for i in tqdm(range(N)):

        batch["time"] = torch.ones((batch_size,), device=device) * i / N

        pred = model(batch)
        batch["input"] = batch["input"].detach().clone() + pred * dt


    print("output min max", batch["input"].min(), batch["input"].max())
    out = batch["input"].cpu() #B, C, H, W
    out = (out + 1) * 0.5  # Convert from [-1,1] to [0,1]
    out = out.clamp(0, 1)
    out = out.permute(0, 2, 3, 1) 
    out = (out * 255).to(torch.uint8)
    
    # Limit to first 64 images and create 8x8 grid
    num_images = min(64, batch_size)
    rows, cols = 8, 8
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    
    # Convert tensor to numpy for plotting
    images = out[:num_images].numpy()
    class_indices = class_indices[:num_images]
    
    # Plot each image
    for idx in range(rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        if idx < len(images):
            ax.imshow(images[idx])
            if classes is not None:
                class_name = classes[class_indices[idx].item()]
                ax.text(0.5, -0.1, class_name, 
                       horizontalalignment='center',
                       transform=ax.transAxes)
        else:
            ax.axis('off')  # Hide empty subplots
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"rf_sample_euler_{N}.png", bbox_inches='tight', dpi=300)
    plt.close()


@torch.no_grad()
def rf_sample_euler_cfg(model: torch.nn.Module, 
                    N: int = 100, 
                    batch_size: int = 16, 
                    device: str = "cuda",
                    classes: List[str] = None,
                    cfg_scales: List[float] = [2.0],
                    batch_dtype = torch.bfloat16
                    ):

    # create a dummy batch
    class_indices = torch.randint(0, model.num_classes, (batch_size,), device=device)
    batch = {
        "input": torch.randn(batch_size, model.input_channels, model.input_image_size[0], model.input_image_size[1], device=device, dtype=batch_dtype),
        "class_idx": class_indices,
    }

    # Create two sets of outputs for different CFG scales
    outputs = []
    dt = 1. / N
    
    for cfg_scale in cfg_scales:
        current_batch = {
            "input": batch["input"].clone(),
            "class_idx": batch["class_idx"].clone(),
        }
        
        for i in tqdm(range(N)):
            current_batch["time"] = torch.ones((batch_size,), device=device, dtype = batch_dtype) * i / N

            # Run both conditional and unconditional forward passes
            if cfg_scale > 0:
                # Conditional
                cond_pred = model(current_batch)
                # Unconditional (using force_drop_ids=1 to force classifier-free)
                current_batch["force_drop_ids"] = torch.ones_like(current_batch["class_idx"])
                uncond_pred = model(current_batch)
                current_batch.pop("force_drop_ids")
                
                # Apply CFG
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            else:
                # Just run normal conditional
                pred = model(current_batch)
                
            current_batch["input"] = current_batch["input"].detach().clone() + pred * dt
        
        outputs.append(current_batch["input"])

    # Process outputs
    print("output min max", outputs[0].min(), outputs[0].max())
    # Concatenate the outputs horizontally
    combined_out = torch.cat(outputs, dim=3)  # Concatenate along width
    out = combined_out.cpu()
    out = (out + 1) * 0.5  # Convert from [-1,1] to [0,1]
    out = out.clamp(0, 1)
    out = out.permute(0, 2, 3, 1)
    out = (out * 255).to(torch.uint8)
    
    # Limit to first 64 images and create 8x8 grid
    num_images = min(64, batch_size)
    cols = batch_size // 8
    rows = batch_size // cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3 * len(cfg_scales), rows * 3))  # Scale width by number of cfg_scales
    
    # Convert tensor to numpy for plotting
    images = out[:num_images].numpy()
    class_indices = class_indices[:num_images]
    
    # Plot each image
    for idx in range(rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        if idx < len(images):
            ax.imshow(images[idx])
            if classes is not None:
                class_name = classes[class_indices[idx].item()]
                ax.text(0.5, -0.1, class_name, 
                       horizontalalignment='center',
                       transform=ax.transAxes)
        else:
            ax.axis('off')
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"rf_sample_euler_cfg_comparison_{N}.png", bbox_inches='tight', dpi=300)
    plt.close()
    return fig