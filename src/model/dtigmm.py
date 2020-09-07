import torch
import torch.nn as nn
import numpy as np

from .gmm import GaussianMixtureModel
from .transformer import PrototypeTransformationNetwork
from utils.logger import print_warning


NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2


class DTIGMM(nn.Module):
    name = 'dtigmm'

    def __init__(self, dataset=None, n_prototypes=10, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        self.n_prototypes = n_prototypes
        self.gmm = GaussianMixtureModel(dataset, n_prototypes, **kwargs)
        self.transformer = PrototypeTransformationNetwork(dataset.n_channels, dataset.img_size, n_prototypes, **kwargs)
        self.empty_cluster_threshold = kwargs.get('empty_cluster_threshold', EMPTY_CLUSTER_THRESHOLD / n_prototypes)
        self._reassign_cluster = kwargs.get('reassign_cluster', True)

    def forward(self, x):
        return self.gmm(x, transformer=self.transformer)

    @property
    def prototypes(self):
        return self.gmm.mus

    @property
    def sigmas(self):
        return self.gmm.sigmas

    @property
    def variances(self):
        return self.gmm.sigmas.detach()**2 + self.gmm.var_min

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        else:
            mus = self.gmm.mus.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            return self.transformer(x, mus)[1]

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster:
            return [], 0

        idx = np.argmax(proportions)
        reassigned = []
        for i in range(self.n_prototypes):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j):
        self.gmm.restart_branch_from(i, j, NOISE_SCALE)
        self.transformer.restart_branch_from(i, j, noise_scale=0)
