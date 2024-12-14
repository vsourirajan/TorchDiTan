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

class InfiniteDataLoader:
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler
        self.iterator = iter(dataloader)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            if self.sampler is not None:
                self.sampler.set_epoch(self.sampler.epoch + 1)
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

def get_cifar10_dataloader(root_dir, num_workers, image_size, batch_size):
    print("building cifar10 dataloader")
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
    infinite_loader = InfiniteDataLoader(data_loader, sampler)
    return infinite_loader, sampler, dataset.classes


if __name__ == "__main__":
    
    def get_cifar10_dataloader_single(root_dir, num_workers, image_size, batch_size):
        print("building cifar10 dataloader")
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
        ])
        base_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
        dataset = CIFAR10Wrapper(base_dataset)
        data_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            shuffle=True,  # Add shuffle=True since we removed the sampler
                            pin_memory=True)
        return data_loader, None, dataset.classes

    dataloader, sampler, classes = get_cifar10_dataloader_single("/local/vondrick/alper/", 1, 32, 1)
    print(dataloader)
    print(sampler)
    print(classes)

    for batch in dataloader:
        print(batch)
        break

    