import os
import torch

import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}


class TinyImagenet(VisionDataset):
    """ TinyImagenet Dataset.
    Note: We download TinyImagenet dataset from <http://cs231n.stanford.edu/tiny-imagenet-200.zip>, then repack it as `.npz` format. 

    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'val'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/train.npz",
            "train.npz",
            "db414016436353892fdf00cb30b9ee57",
        ],
        "val": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/val.npz",
            "val.npz",
            "7762694b6217fec8ba1a7be3c20ef218",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # reading(loading) npz file as array
        loaded_npz = np.load(os.path.join(self.root, self.filename))
        self.data = loaded_npz['image']
        self.targets = loaded_npz["label"].tolist()
        print(split+' images size:', self.data.shape)
        print(split+' labels size:', len(self.targets))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        print(fpath)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)



def load_tinyimagenet(data_dir, use_augmentation='base'):
    """
    Returns Tiny Imagenet-200 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = TinyImagenet(root=data_dir, split='train', download=True, transform=train_transform)
    test_dataset = TinyImagenet(root=data_dir, split='val', download=True, transform=test_transform)

    return train_dataset, test_dataset
