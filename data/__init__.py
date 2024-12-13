from data.imagenet import get_imagenet_dataloader   
from data.cifar10 import get_cifar10_dataloader

def build_image_dataloader(dataset_config):
    root_dir = dataset_config.root_dir
    num_workers = dataset_config.num_workers
    image_size = dataset_config.image_size
    batch_size = dataset_config.batch_size

    if dataset_config.dataset_name == "imagenet":
        return get_imagenet_dataloader(root_dir, num_workers, image_size, batch_size)
    elif dataset_config.dataset_name == "cifar10":
        return get_cifar10_dataloader(root_dir, num_workers, image_size, batch_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config.dataset_name}") 
