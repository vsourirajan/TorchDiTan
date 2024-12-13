import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageNet
from typing import Optional, Callable, Dict

class ImageNetDataset(Dataset):
    """
    A wrapper around torchvision's ImageNet dataset.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            root (str): Root directory of the ImageNet dataset
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied to images
        """
        self.dataset = ImageNet(
            root=root,
            split=split,
            transform=transform
        )
        # Get the idx to class mapping from torchvision's ImageNet
        self.idx_to_class = {idx: cls for cls, idx in self.dataset.class_to_idx.items()}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        return {
            "original_input": image,  # Image tensor
            "class_idx": class_idx,  # Class index
        }
    
    def get_class_name(self, idx: int) -> str:
        """
        Get class name from class index.
        
        Args:
            idx (int): Class index
            
        Returns:
            str: Class name
        """
        return self.idx_to_class[idx]
    
    @property
    def classes(self) -> Dict[int, str]:
        """
        Get dictionary mapping class indices to class names.
        
        Returns:
            Dict[int, str]: Dictionary mapping class indices to class names
        """
        return self.idx_to_class

def get_imagenet_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 4,
    image_size: int = 256,
    transform: Optional[Callable] = None,
):
    """
    Creates a DataLoader for ImageNet dataset using torchvision's ImageNet.
    
    Args:
        root_dir (str): Root directory of ImageNet dataset
        split (str): 'train' or 'val'
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of worker processes for data loading
        transform (callable, optional): Optional transform to be applied to images
        
    Returns:
        DataLoader: PyTorch DataLoader for ImageNet
    """
    # Default ImageNet normalization if no transform is provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    dataset = ImageNetDataset(
        root=root_dir,
        split=split,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    sampler = DistributedSampler(dataset)

    return dataloader, sampler

if __name__ == "__main__":
    # Set the root directory for ImageNet
    root_dir = "/local/vondrick/datasets/imagenet"
    
    # Example custom transform
    custom_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    # Create train and validation dataloaders
    train_loader, _ = get_imagenet_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=32,
        transform=custom_transform
    )
    
    val_loader, _ = get_imagenet_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=32,
        transform=custom_transform
    )
    
    # Example usage
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Get a batch of data
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Example of class name lookup
    dataset = train_loader.dataset
    for i in range(5):  # Print first 5 class names
        class_name = dataset.get_class_name(i)
        print(f"Class {i}: {class_name}")
