from collections import OrderedDict
from scipy import signal

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn.cluster import KMeans


def copy_with_noise(t, noise_scale=0.0001):
    return t.detach().clone() + torch.randn(t.shape, device=t.device) * noise_scale


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def create_gaussian_weights(img_size, n_channels, std=10):
    g1d_h = signal.gaussian(img_size[0], std)
    g1d_w = signal.gaussian(img_size[1], std)
    g2d = np.outer(g1d_h, g1d_w)
    return torch.from_numpy(g2d).unsqueeze(0).expand(n_channels, -1, -1)


def generate_data(dataset, K, init_type='sample', value=None):
    samples = []
    if init_type == 'kmeans':
        N = min(100 * K, len(dataset))
        images = next(iter(DataLoader(dataset, batch_size=N, shuffle=True, num_workers=4)))[0]
        img_size = images.shape[1:]
        X = images.flatten(1).numpy()
        cluster = KMeans(K)
        cluster.fit(X)
        samples = list(map(lambda c: torch.Tensor(c).reshape(img_size), cluster.cluster_centers_))
    else:
        for _ in range(K):
            if init_type == 'soup':
                sample = torch.rand(dataset[0][0].shape)
            elif init_type == 'sample':
                sample = dataset[np.random.randint(len(dataset))][0]
            elif init_type == 'constant':
                if value is not None:
                    sample = torch.full(dataset[0][0].shape, value, dtype=torch.float)
                else:
                    raise ValueError("value arg is mandatory with init_type=='constant'")
            else:
                raise NotImplementedError
            samples.append(sample)
    return samples


def safe_model_state_dict(state_dict):
    """
    Converts a state dict saved from a DataParallel module to normal module state_dict
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v  # remove 'module.' prefix
    return new_state_dict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_nb_out_channels(layer):
    return list(filter(lambda e: isinstance(e, nn.Conv2d), layer.modules()))[-1].out_channels


def get_output_size(in_channels, img_size, model):
    x = torch.zeros(1, in_channels, *img_size)
    return np.prod(model(x).shape)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class TPSGrid(nn.Module):
    """Original implem: https://github.com/WarBean/tps_stn_pytorch"""

    def __init__(self, img_size, target_control_points):
        super().__init__()
        img_height, img_width = img_size
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = img_height * img_width
        y, x = torch.meshgrid(torch.linspace(-1, 1, img_height), torch.linspace(-1, 1, img_width))
        target_coordinate = torch.stack([x.flatten(), y.flatten()], 1)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    @staticmethod
    def compute_partial_repr(input_points, control_points):
        """Compute radial basis kernel phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2"""
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        repr_matrix.masked_fill_(repr_matrix != repr_matrix, 0)
        return repr_matrix

    def forward(self, source_control_points):
        Y = torch.cat([source_control_points, self.padding_matrix.expand(source_control_points.size(0), 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        return source_coordinate
