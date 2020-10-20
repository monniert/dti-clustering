from abc import ABCMeta
from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from utils import coerce_to_path_and_check_exist, get_files_from_dir
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH


class _AbstractCollectionDataset(TorchDataset):
    """Abstract torch dataset from raw files collections associated to tags."""
    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3

    def __init__(self, split, img_size, tag, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        self.split = split
        self.tag = tag
        try:
            input_files = get_files_from_dir(self.data_path, IMG_EXTENSIONS, sort=True)
        except FileNotFoundError:
            input_files = []
        self.input_files = input_files
        self.labels = [-1] * len(input_files)
        self.n_classes = 0
        self.size = len(self.input_files)

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.crop = True
        else:
            assert len(img_size) == 2
            self.img_size = img_size
            self.crop = False

        if self.size > 0:
            sample_size = Image.open(self.input_files[0]).size
            if min(self.img_size) > min(sample_size):
                raise ValueError("img_size too big compared to a sampled image size, adjust it or upscale dataset")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        return inp, self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)


class InstagramDataset(_AbstractCollectionDataset):
    name = 'instagram'


class MegaDepthDataset(_AbstractCollectionDataset):
    name = 'megadepth'
