import torch
from torch.nn import functional as F

from utils import use_seed


class ColorAugment():
    def __call__(self, img, seed=None):
        self.apply(img)

    @staticmethod
    def apply(img, seed=None):
        if seed is not None:
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
        return F.interpolate(img.unsqueeze(0), self.img_size, mode='bilinear')[0]

    def __repr__(self):
        return self.__class__.__name__ + '()'
