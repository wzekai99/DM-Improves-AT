import os
import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}


def load_tinyimagenet(data_dir, use_augmentation=False):
    """
    Returns Tiny Imagenet-200 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

    return train_dataset, test_dataset
