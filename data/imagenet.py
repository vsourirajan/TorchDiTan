import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageNet
from typing import Optional, Callable, Dict
from data.imagenet_utils import imagenet_idx_to_class
from datasets import load_dataset
from PIL import Image
import io
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class ImageNetDataset(Dataset):
    """
    A wrapper around ImageNet dataset with support for both local and streaming modes.
    """
    def __init__(
        self,
        root: str = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        streaming: bool = True,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        buffer_size: int = 10_000
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            root (str, optional): Root directory of the ImageNet dataset for local mode
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied to images
            streaming (bool): Whether to use streaming mode from HuggingFace
            shuffle (bool): Whether to shuffle the streaming dataset
            shuffle_seed (int): Seed for shuffling
            buffer_size (int): Buffer size for shuffling streaming data
        """
        self.transform = transform
        self.streaming = streaming
        
        if streaming:
            self.dataset = load_dataset(
                'ILSVRC/imagenet-1k',
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            if shuffle:
                self.dataset = self.dataset.shuffle(
                    seed=shuffle_seed,
                    buffer_size=buffer_size
                )
            self.iterator = iter(self.dataset)
        else:
            self.dataset = ImageNet(
                root=root,
                split=split,
                transform=transform
            )
        
        # Get the idx to class mapping
        self.idx_to_class = imagenet_idx_to_class
    
    def __len__(self):
        if self.streaming:
            return int(1e9)  # Practically infinite for streaming
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.streaming:
            try:
                sample = next(self.iterator)
                # Convert image bytes to PIL Image
                image = sample['image'].convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return {
                    "original_input": image,
                    "class_idx": sample['label']
                }
            except StopIteration:
                # Restart iterator if we reach the end
                self.iterator = iter(self.dataset)
                return self.__getitem__(0)
        else:
            image, class_idx = self.dataset[idx]
            return {
                "original_input": image,
                "class_idx": class_idx,
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
    root_dir: str = None,
    num_workers: int = 4,
    image_size: int = 256,
    batch_size: int = 1,
    streaming: bool = False,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    buffer_size: int = 10_000
):
    """
    Creates a DataLoader for ImageNet dataset.
    
    Args:
        root_dir (str, optional): Root directory of ImageNet dataset for local mode
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size of the images
        batch_size (int): Batch size for the dataloader
        streaming (bool): Whether to use streaming mode
        shuffle (bool): Whether to shuffle the dataset
        shuffle_seed (int): Seed for shuffling
        buffer_size (int): Buffer size for shuffling streaming data
        
    Returns:
        tuple: (DataLoader, Sampler, classes_dict)
    """
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
        streaming=streaming,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed + local_rank,  # Add rank to seed
        buffer_size=buffer_size
    )
    
    if streaming:
        # For streaming, we don't use DistributedSampler
        sampler = None
        dataloader_shuffle = False  # Shuffling is handled by the dataset
    else:
        sampler = DistributedSampler(dataset)
        dataloader_shuffle = True if sampler is None else False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )

    return dataloader, sampler, dataset.classes

def test_distributed():
    # Initialize distributed environment
    if dist.is_available():
        if not dist.is_initialized():
            dist.init_process_group("nccl")  # or use "gloo" if no GPU
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Create dataloader for this rank
    dataloader, _, classes = get_imagenet_dataloader(
        root_dir="/local/vondrick/datasets/imagenet",
        streaming=False,
        batch_size=8,
        shuffle=True,
        buffer_size=1000,
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
