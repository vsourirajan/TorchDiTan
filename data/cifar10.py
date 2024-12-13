import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

class CIFAR10Wrapper(torch.utils.data.Dataset):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        return {
            "original_input": image,  # Already [C,H,W] from ToTensor()
            "class_idx": class_idx,
            "class_name": self.classes[class_idx]
        }

#root_dir, num_workers, image_size, batch_size

def get_cifar10_dataloader(root_dir, num_workers, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
    ])
    base_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    dataset = CIFAR10Wrapper(base_dataset)
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                pin_memory=True,
                                sampler=sampler)
    return data_loader, sampler