import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup import SemiSupervisedDataset
from .semisup import SemiSupervisedSampler

from .autoaugment import CIFAR10Policy
from .idbh import IDBH
from RandAugment import RandAugment # pip install git+https://github.com/ildoonet/pytorch-randaugment


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def load_cifar10s(data_dir, use_augmentation='base', use_consistency=False, aux_take_amount=None, 
                  aux_data_filename='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle', 
                  validation=False):
    """
    Returns semisupervised CIFAR10 train, test datasets and dataloaders (with Tiny Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation: use different augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('cifar10s', 'cifar10', data_dir)
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'none':
        train_transform = test_transform
    elif use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    elif use_augmentation == 'cutout':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(18))
    elif use_augmentation == 'autoaugment':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            CIFAR10Policy(),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(18))
    elif use_augmentation == 'randaugment':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
        # Add RandAugment with N, M(hyperparameter), N=2, M=14 for wdn-28-10
        train_transform.transforms.insert(0, RandAugment(2, 14))
    elif use_augmentation == 'idbh':
        train_transform = IDBH('cifar10-weak')
    
    if use_consistency:
        train_transform = MultiDataTransform(train_transform)

    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=True, download=True, 
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=False, download=True, 
                                         transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  # split from training set
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar10', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.dataset_size = len(self.dataset)
