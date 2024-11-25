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

def get_cifar10_dataloader(job_config, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
    ])
    base_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=transform, download=True)
    dataset = CIFAR10Wrapper(base_dataset)
    sampler = DistributedSampler(dataset)
    batch_size_per_gpu = job_config.training.batch_size // world_size
    data_loader = DataLoader(dataset, 
                                batch_size=batch_size_per_gpu, 
                                num_workers=16, 
                                pin_memory=True,
                                sampler=sampler)
    return data_loader, sampler