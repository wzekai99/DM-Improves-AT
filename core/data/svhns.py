import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup import SemiSupervisedDataset, SemiSupervisedDatasetSVHN
from .semisup import SemiSupervisedSampler


def load_svhns(data_dir, use_augmentation='base', use_consistency=False, aux_take_amount=None, 
                  aux_data_filename='/cluster/scratch/rarade/svhns/ti_500K_pseudo_labeled.pickle', 
                  validation=False):
    """
    Returns semisupervised SVHN train, test datasets and dataloaders (with Tiny Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('svhns', 'svhn', data_dir)
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = test_transform
    
    train_dataset = SemiSupervisedSVHN(base_dataset='svhn', root=data_dir, train=True, download=True, 
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedSVHN(base_dataset='svhn', root=data_dir, train=False, download=True, 
                                         transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset


class SemiSupervisedSVHN(SemiSupervisedDatasetSVHN):
    """
    A dataset with auxiliary pseudo-labeled data for SVHN.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'svhn', 'Only semi-supervised SVHN is supported. Please use correct dataset!'
        if train:
            self.dataset = torchvision.datasets.SVHN(split='train', **kwargs)
        else:
            self.dataset = torchvision.datasets.SVHN(split='test', **kwargs)
        self.dataset_size = len(self.dataset)
