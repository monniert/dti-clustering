from abc import ABCMeta
from functools import lru_cache

import h5py
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, ToTensor

from .torch_transforms import TensorResize
from utils import coerce_to_path_and_check_exist, use_seed
from utils.path import DATASETS_PATH

INPUT_EXTENSIONS = ['jpeg', 'jpg', 'JPG', 'png']
HDF5_FILE = 'data4torch.h5'
VAL_SPLIT_RATIO = 0.1


def load_hdf5_file(filename):
    with h5py.File(filename, mode='r') as f:
        data, labels = np.asarray(f['data'], dtype=np.uint8), np.asarray(f['labels'], dtype=np.int64)
    return data, labels


class _AbstractHDF5Dataset(TorchDataset):
    """Abstract torch dataset from HDF5 files."""
    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_classes = NotImplementedError
    n_channels = NotImplementedError
    img_size = NotImplementedError  # Original img_size
    mean = NotImplementedError
    std = NotImplementedError
    label_shift = -1
    transposed = False

    def __init__(self, split, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name / HDF5_FILE)
        self.split = split
        data, labels = load_hdf5_file(self.data_path)
        data = data.swapaxes(1, 3)  # NHWC
        if self.transposed:  # swap H and W axes
            data = data.swapaxes(1, 2)
        unique_labels = sorted(np.unique(labels))
        consecutive_labels = (np.diff(unique_labels) == 1).all()
        if not consecutive_labels:
            for k, l in enumerate(unique_labels, start=1):
                labels[labels == l] = k

        if split == 'val':
            n_val = round(VAL_SPLIT_RATIO * len(data))
            with use_seed(46):
                indices = np.random.choice(range(len(data)), n_val, replace=False)
            data, labels = data[indices], labels[indices]
        self.data, self.labels = data, labels
        self.size = len(self.labels)

        img_size = kwargs.get('img_size')
        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert len(self.img_size) == 2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx] + self.label_shift

    @property
    @lru_cache()
    def transform(self):
        transform = [ToTensor()]
        if self.img_size != self.__class__.img_size:
            transform.append(TensorResize(self.img_size))
        return Compose(transform)


class FRGCDataset(_AbstractHDF5Dataset):
    name = 'FRGC'
    n_classes = 20
    n_channels = 3
    img_size = (32, 32)
