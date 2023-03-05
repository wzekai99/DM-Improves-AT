import torch

import torchvision
import torchvision.transforms as transforms


DATA_DESC = {
    'data': 'cifar100',
    'classes': tuple(range(0, 100)),
    'num_classes': 100,
    'mean': [0.5071, 0.4865, 0.4409], 
    'std': [0.2673, 0.2564, 0.2762],
}


def load_cifar100(data_dir, use_augmentation='base'):
    """
    Returns CIFAR100 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.RandomRotation(15), transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset