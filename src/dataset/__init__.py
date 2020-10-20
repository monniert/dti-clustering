from .custom import AffNISTTestDataset
from .hdf5 import FRGCDataset
from .raw import InstagramDataset, MegaDepthDataset
from .torchvision import (FashionMNISTDataset, MNISTDataset, MNISTTestDataset, MNISTColorDataset, MNIST1kDataset,
                          SVHNDataset, USPSDataset)


def get_dataset(dataset_name):
    return {
        # Custom
        'affnist_test': AffNISTTestDataset,

        # HDF5
        'frgc': FRGCDataset,

        # Raw
        'instagram': InstagramDataset,
        'megadepth': MegaDepthDataset,

        # Torchvision
        'fashion_mnist': FashionMNISTDataset,
        'mnist': MNISTDataset,
        'mnist_test': MNISTTestDataset,
        'mnist_color': MNISTColorDataset,
        'mnist_1k': MNIST1kDataset,
        'svhn': SVHNDataset,
        'usps': USPSDataset,
    }[dataset_name]
