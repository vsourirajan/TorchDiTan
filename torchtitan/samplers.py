import torch
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb

def visualize_samples(images, class_indices, classes, rows, cols, title=None, cfg_scales=None, filename_prefix="rf_sample"):
    """
    Visualize images in a grid layout, save locally and return PIL image
    
    Args:
        images: Tensor of shape [B, H, W, C] in uint8 format (0-255)
        class_indices: Tensor of class indices
        classes: List of class names
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        title: Optional title for the figure
        cfg_scales: Optional list of cfg scales used (for filename)
        filename_prefix: Prefix for the saved file
    
    Returns:
        PIL.Image: Image in PIL format
    """

    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided for visualization")

    # Create figure with subplots
    fig_width = cols * 3 * (len(cfg_scales) if cfg_scales else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, rows * 3))
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    # Convert axes to 2D array if it's 1D
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each image
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            
            if idx < num_images:
                ax.imshow(images[idx])
                if classes is not None and class_indices is not None:
                    class_name = classes[class_indices[idx].item()]
                    ax.text(0.5, -0.1, class_name[:20], 
                           horizontalalignment='center',
                           transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx >= num_images:
                ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    
    # Convert to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert to PIL Image
    pil_image = Image.fromarray(data)
    
    # Create filename with cfg scales if provided
    if cfg_scales:
        cfg_str = "_".join([f"cfg{scale}" for scale in cfg_scales])
        filename = f"{filename_prefix}_{cfg_str}.png"
    else:
        filename = f"{filename_prefix}.png"
    
    # Save the image
    pil_image.save(filename)
    
    plt.close(fig)  # Close the figure to free memory
    
    return pil_image


@torch.no_grad()
def rf_sample_euler_cfg(model: torch.nn.Module, 
                    N: int = 100, 
                    batch_size: int = 16, 
                    device: str = "cuda",
                    classes: List[str] = None,
                    cfg_scales: List[float] = [3.0],
                    batch_dtype = torch.bfloat16,
                    class_indices = None
                    ):

    # Use provided class indices or generate random ones
    if class_indices is None:
        class_indices = torch.randint(0, model.num_classes, (batch_size,), device=device)
    else:
        class_indices = class_indices.to(device)

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
            if cfg_scale > 1.0:
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
    print("neural net out min max", out.min(), out.max(), "mean", out.mean())
    out = (out + 1) * 0.5  # Convert from [-1,1] to [0,1]
    out = out.clamp(0, 1)
    out = out.permute(0, 2, 3, 1)
    out = (out * 255).to(torch.uint8)
    
    # Return all generated images, not just first 16
    return out.numpy()

@torch.no_grad()
def sample_and_visualize(model, batch, param_dtype, classes, num_vis_samples=16):
    """
    Generate and visualize both real and generated samples.
    
    Args:
        model: The model to generate samples with
        batch: Dictionary containing real images and class indices
        param_dtype: Data type for model parameters
        classes: List of class names
        num_vis_samples: Number of samples to visualize (default: 16)
    
    Returns:
        dict: Dictionary containing wandb Image objects for real and generated images
    """

    #make sure we don't try to visualize more images than there are in a training batch
    #otherwise we'll get an oom
    num_vis_samples = min(num_vis_samples, batch["original_input"].shape[0])

    # Process real images
    real_images = batch["original_input"][:num_vis_samples]
    real_classes = batch["class_idx"][:num_vis_samples]
    
    # Convert real images to visualization format
    real_vis = (real_images.cpu() + 1) * 0.5
    real_vis = real_vis.clamp(0, 1)
    real_vis = real_vis.permute(0, 2, 3, 1)
    real_vis = (real_vis * 255).to(torch.uint8)
    real_images_array = real_vis.numpy()

    # Generate samples
    generated_images_array = rf_sample_euler_cfg(
        model, 
        N=32, 
        batch_size=num_vis_samples,
        device="cuda", 
        classes=classes, 
        batch_dtype=param_dtype,
        class_indices=real_classes
    )

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_vis_samples)))
    
    # Create image grids
    real_grid = visualize_samples(
        images=real_images_array,
        class_indices=real_classes,
        classes=classes,
        rows=grid_size,
        cols=grid_size,
        title="Real Images",
        filename_prefix='out_images_real'
    )
    
    gen_grid = visualize_samples(
        images=generated_images_array,
        class_indices=real_classes,
        classes=classes,
        rows=grid_size,
        cols=grid_size,
        title="Generated Images",
        filename_prefix='out_images_gen'
    )

    return {
        "real_images": wandb.Image(real_grid),
        "generated_images": wandb.Image(gen_grid)
    }