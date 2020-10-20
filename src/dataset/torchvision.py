from abc import ABCMeta

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset, ConcatDataset
from torchvision.datasets import FashionMNIST, MNIST, SVHN, USPS
from torchvision.transforms import ToTensor, Compose

from utils import use_seed
from utils.path import DATASETS_PATH
from .torch_transforms import ColorAugment


VAL_SPLIT_RATIO = 0.1


class _AbstractTorchvisionDataset(TorchDataset):
    """_Abstract torchvision dataset"""
    __metaclass__ = ABCMeta
    root = DATASETS_PATH

    dataset_class = NotImplementedError
    name = NotImplementedError
    n_classes = NotImplementedError
    n_channels = NotImplementedError
    img_size = NotImplementedError
    test_split_only = False
    label_shift = 0
    n_samples = None

    def __init__(self, split, **kwargs):
        super().__init__()
        self.split = split
        self.eval_mode = kwargs.get('eval_mode', False)

        kwargs = {}
        if self.name in ['svhn']:
            kwargs['split'] = 'test'
        else:
            kwargs['train'] = False
        dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, **kwargs)
        if self.n_samples is not None:
            assert self.n_samples < len(dataset)
            with use_seed(46):
                indices = np.random.choice(range(len(dataset)), self.n_samples, replace=False)
            dataset.data = dataset.data[indices]
            dataset.targets = dataset.targets[indices] if hasattr(dataset, 'targets') else dataset.labels[indices]

        if split == 'val':
            n_val = max(round(VAL_SPLIT_RATIO * len(dataset)), 100)
            if n_val < len(dataset):
                with use_seed(46):
                    indices = np.random.choice(range(len(dataset)), n_val, replace=False)

                dataset.data = dataset.data[indices]
                if hasattr(dataset, 'targets'):
                    dataset.targets = np.asarray(dataset.targets)[indices]
                else:
                    dataset.labels = np.asarray(dataset.labels)[indices]
        elif not self.test_split_only and self.n_samples is None:
            kwargs = {}
            if self.name in ['svhn']:
                kwargs['split'] = 'train'
            else:
                kwargs['train'] = True
            train_dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, **kwargs)
            sets = [dataset, train_dataset]
            if self.name == 'svhn' and not self.eval_mode:
                sets.append(self.dataset_class(root=self.root, transform=self.transform, download=True, split='extra'))
            dataset = ConcatDataset(sets)

        self.dataset = dataset

    @property
    def transform(self):
        return Compose([ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label + self.label_shift


class FashionMNISTDataset(_AbstractTorchvisionDataset):
    dataset_class = FashionMNIST
    name = 'fashion_mnist'
    n_classes = 10
    n_channels = 1
    img_size = (28, 28)


class MNISTDataset(_AbstractTorchvisionDataset):
    dataset_class = MNIST
    name = 'mnist'
    n_classes = 10
    n_channels = 1
    img_size = (28, 28)


class MNISTTestDataset(MNISTDataset):
    name = 'mnist_test'
    test_split_only = True


class MNIST1kDataset(MNISTDataset):
    name = 'mnist_1k'
    test_split_only = True
    n_samples = 1000


class MNISTColorDataset(MNISTDataset):
    name = 'mnist_color'
    n_channels = 3

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = ColorAugment.apply(img, seed=idx)
        return img, label + self.label_shift


class SVHNDataset(_AbstractTorchvisionDataset):
    dataset_class = SVHN
    name = 'svhn'
    n_classes = 10
    n_channels = 3
    img_size = (32, 32)


class USPSDataset(_AbstractTorchvisionDataset):
    dataset_class = USPS
    name = 'usps'
    n_classes = 10
    n_channels = 1
    img_size = (16, 16)
