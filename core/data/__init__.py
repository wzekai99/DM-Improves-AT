import os
import torch

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .svhn import load_svhn
from .cifar10s import load_cifar10s
from .cifar100s import load_cifar100s
from .svhns import load_svhns
from .tiny_imagenet import load_tinyimagenet
from .tiny_imagenets import load_tinyimagenets

from .semisup import get_semisup_dataloaders


SEMISUP_DATASETS = ['cifar10s', 'cifar100s', 'svhns', 'tiny-imagenets']
DATASETS = ['cifar10', 'svhn', 'cifar100', 'tiny-imagenet'] + SEMISUP_DATASETS

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
    'cifar100': load_cifar100,
    'svhn': load_svhn,
    'tiny-imagenet': load_tinyimagenet,
    'cifar10s': load_cifar10s,
    'cifar100s': load_cifar100s,
    'svhns': load_svhns,
    'tiny-imagenets': load_tinyimagenets,
}


def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar100' in data_dir:
        from .cifar100 import DATA_DESC
    elif 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    elif 'svhn' in data_dir:
        from .svhn import DATA_DESC
    elif 'tiny-imagenet' in data_dir:
        from .tiny_imagenet import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_dir, batch_size=256, batch_size_test=256, num_workers=4, use_augmentation='base', use_consistency=False, shuffle_train=True, 
              aux_data_filename=None, unsup_fraction=None, validation=False):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (base/none): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]
    
    if validation:
        assert dataset in SEMISUP_DATASETS, 'Only semi-supervised datasets allow a validation set.'
        train_dataset, test_dataset, val_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation, use_consistency=use_consistency,
                                                                   aux_data_filename=aux_data_filename, validation=True)
    else:
        train_dataset, test_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation)
       
    if dataset in SEMISUP_DATASETS:
        if validation:
            train_dataloader, test_dataloader, val_dataloader = get_semisup_dataloaders(
                train_dataset, test_dataset, val_dataset, batch_size, batch_size_test, num_workers, unsup_fraction
            )
        else:
            train_dataloader, test_dataloader = get_semisup_dataloaders(
                train_dataset, test_dataset, None, batch_size, batch_size_test, num_workers, unsup_fraction
            )
    else:
        #pin_memory = torch.cuda.is_available()
        pin_memory = False
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                       num_workers=num_workers, pin_memory=pin_memory)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                      num_workers=num_workers, pin_memory=pin_memory)
    if validation:
        return train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader
    return train_dataset, test_dataset, train_dataloader, test_dataloader
