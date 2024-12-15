import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageNet
from typing import Optional, Callable, Dict
from data.imagenet_utils import imagenet_idx_to_class
import time
import torch.distributed as dist
from safetensors.torch import safe_open
from tqdm import tqdm
import glob
from safetensors import safe_open
from safetensors.torch import save_file

class ImageNetDataset(Dataset):
    """
    A wrapper around ImageNet dataset that can return either pixel images or latent representations.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        mode: str = 'latents',  # 'pixels' or 'latents'
        tokenizer_type: str = 'continuous'  # 'continuous' or 'discrete'
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            root (str): Root directory of the ImageNet dataset or path to safetensors file
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied to images
            mode (str): 'pixels' or 'latents'
            tokenizer_type (str): 'continuous' or 'discrete', only used if mode='latents'
        """
        self.mode = mode
        self.transform = transform
        self.root = root  # Store the root path
        self.tokenizer_type = tokenizer_type
        
        if mode == 'pixels':
            self.dataset = ImageNet(
                root=root,
                split=split,
                transform=transform
            )
            # Get the idx to class mapping
            self.idx_to_class = imagenet_idx_to_class
            
        else:  # mode == 'latents'
            # Get rank information
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            
            # Load only this rank's portion of the data
            with safe_open(root, framework="pt", device="cpu") as f:
                if self.tokenizer_type == "continuous":
                    self.data = f.get_tensor("latents").to(torch.bfloat16) * 16.0 / 255.0
                else:
                    self.data = f.get_tensor("indices").to(torch.uint16)
                self.labels = f.get_tensor("labels")
        
        self.idx_to_class = imagenet_idx_to_class

    def __len__(self):
        if self.mode == 'pixels':
            return len(self.dataset)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'pixels':
            image, class_idx = self.dataset[idx]
            return {
                "original_input": image,
                "class_idx": class_idx,
            }
        else:
            return {
                "original_input": self.data[idx],
                "class_idx": self.labels[idx].item(),
            }
    
    def get_class_name(self, idx: int) -> str:
        """
        Get class name from class index.
        """
        return self.idx_to_class[idx]
    
    @property
    def classes(self) -> Dict[int, str]:
        """
        Get dictionary mapping class indices to class names.
        """
        return self.idx_to_class


def get_imagenet_dataloader(
    root_dir: str,
    num_workers: int = 8,
    image_size: int = 256,
    batch_size: int = 1,
    shuffle: bool = True,
    mode: str = 'latents',
    tokenizer_type: str = 'continuous',
):
    """
    Creates a DataLoader for ImageNet dataset.
    
    Args:
        root_dir (str): Root directory of ImageNet dataset or path to safetensors file
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size of the images (only used if mode='pixels')
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the dataset
        mode (str): 'pixels' or 'latents'
        tokenizer_type (str): 'continuous' or 'discrete', only used if mode='latents'
        
    Returns:
        tuple: (DataLoader, Sampler, classes_dict)
    """
    transform = None
    if mode == 'pixels':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
        
    # Get local rank for seed modification
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    dataset = ImageNetDataset(
        root=root_dir,
        split='train',
        transform=transform,
        mode=mode,
        tokenizer_type=tokenizer_type
    )
    
    sampler = DistributedSampler(dataset)
    dataloader_shuffle = True if sampler is None else False
    
    print("[INFO] Creating dataloader with {} workers".format(num_workers), flush=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return dataloader, sampler, dataset.classes

def test_distributed(root_dir: str):
    # Initialize distributed environment
    if dist.is_available():
        if not dist.is_initialized():
            dist.init_process_group("nccl")  # or use "gloo" if no GPU
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Create dataloader for this rank
    dataloader, _, classes = get_imagenet_dataloader(
        root_dir=root_dir,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # Test different samples across ranks
    print(f"\n=== Testing different samples across ranks ===")
    first_batch = next(iter(dataloader))
    class_indices = first_batch['class_idx']
    print(f"Rank {local_rank}: first batch classes: {class_indices}")
    
    # Measure throughput
    print(f"\n=== Measuring throughput on rank {local_rank} ===")
    num_batches = 100
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        print("batch images shape", batch['original_input'].shape)
        
        if i >= num_batches:
            break
        
        if i % 10 == 0:
            print(f"Rank {local_rank}: Processing batch {i}/{num_batches}")
    
    end_time = time.time()
    duration = end_time - start_time
    images_per_second = (num_batches * dataloader.batch_size) / duration
    
    print(f"\nRank {local_rank} Statistics:")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Images per second: {images_per_second:.2f}")
    print(f"Batches per second: {num_batches/duration:.2f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":

    # torchrun \
    # --nproc_per_node=2 \
    # --master_port=29500 \
    # data/imagenet.py \
    # --root_dir=/path/to/your/imagenet/dataset

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/local/vondrick/datasets/imagenet")
    args = parser.parse_args()
    test_distributed(root_dir=args.root_dir)
