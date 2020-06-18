import torch
from torch.nn import functional as F

from utils import use_seed


class ColorAugment():
    def __init__(self, use_seed=False):
        self.use_seed = use_seed

    def __call__(self, img):
        if self.use_seed:
            seed = int(img.sum().item() * 1e6)
            with use_seed(seed):
                color = torch.rand(3, 1, 1) * 2 - 1
                bias = torch.rand(3, 1, 1)
        else:
            color = torch.rand(3, 1, 1) * 2 - 1
            bias = torch.rand(3, 1, 1)
        img = torch.abs(color * img + bias)
        return img / img.max()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TensorResize():
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        # XXX interpolate first dim is a batch dim
        return F.interpolate(img.unsqueeze(0), self.img_size)[0]

    def __repr__(self):
        return self.__class__.__name__ + '()'
