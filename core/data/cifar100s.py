import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup import SemiSupervisedDataset


def load_cifar100s(data_dir, use_augmentation='base', use_consistency=False, aux_take_amount=None, 
                   aux_data_filename=None, validation=False):
    """
    Returns semisupervised CIFAR100 train, test datasets and dataloaders (with DDPM Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('cifar100s', 'cifar100', data_dir)
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.RandomRotation(15), transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = SemiSupervisedCIFAR100(base_dataset='cifar100', root=data_dir, train=True, download=True, 
                                           transform=train_transform, aux_data_filename=aux_data_filename, 
                                           add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedCIFAR100(base_dataset='cifar100', root=data_dir, train=False, download=True, 
                                          transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset, None


class SemiSupervisedCIFAR100(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR100.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar100', 'Only semi-supervised cifar100 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR100(train=train, **kwargs)
        self.dataset_size = len(self.dataset)