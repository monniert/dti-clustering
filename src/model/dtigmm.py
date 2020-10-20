import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans

from .transformer import PrototypeTransformationNetwork
from .tools import copy_with_noise, create_gaussian_weights, generate_data
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

        # GMM initialization
        init_type = kwargs.get('init_type', 'sample')
        sigma = kwargs.get('sigma_init', 0.5)
        var_min = kwargs.get('var_min', 0.25**2)
        if init_type == 'kmeans':
            mus, sigmas = self._kmeans_init(dataset, n_prototypes, var_min)
        else:
            mus = generate_data(dataset, K=n_prototypes, init_type=init_type)
            sigmas = generate_data(dataset, K=n_prototypes, init_type='constant', value=sigma)

        self.mus = nn.Parameter(torch.stack(mus))
        self.sigmas = nn.Parameter(torch.stack(sigmas))
        self.var_min = var_min
        self.n = mus[0].numel()
        self.register_buffer('const', torch.Tensor([2*np.pi]))
        self.mixing_params = nn.Parameter(torch.ones(n_prototypes))

        # Other initialization
        self.transformer = PrototypeTransformationNetwork(dataset.n_channels, dataset.img_size, n_prototypes, **kwargs)
        self.empty_cluster_threshold = kwargs.get('empty_cluster_threshold', EMPTY_CLUSTER_THRESHOLD / n_prototypes)
        self._reassign_cluster = kwargs.get('reassign_cluster', True)
        use_gaussian_weights = kwargs.get('gaussian_weights', False)
        if use_gaussian_weights:
            std = kwargs['gaussian_weights_std']
            self.register_buffer('loss_weights', create_gaussian_weights(dataset.img_size, dataset.n_channels, std))
        else:
            self.loss_weights = None

    @staticmethod
    def _kmeans_init(dataset, n_clusters, var_min=0.1):
        images = next(iter(DataLoader(dataset, batch_size=100*n_clusters, shuffle=True, num_workers=4)))[0]
        img_size = images.shape[1:]
        X = images.flatten(1).numpy()
        cluster = KMeans(n_clusters)
        labels = cluster.fit_predict(X)
        mus = list(map(lambda c: torch.Tensor(c).reshape(img_size), cluster.cluster_centers_))
        sigmas = []
        for k in range(n_clusters):
            x = X[labels == k]
            s = np.sqrt((np.sum((x - cluster.cluster_centers_[k])**2, axis=0) / (len(x) - 1) - var_min).clip(0))
            sigmas.append(torch.Tensor(s).reshape(img_size))
        return mus, sigmas

    @property
    def prototypes(self):
        return self.mus

    @property
    def variances(self):
        return self.sigmas**2 + self.var_min

    def cluster_parameters(self):
        return [self.mus, self.sigmas, self.mixing_params]

    def transformer_parameters(self):
        return self.transformer.parameters()

    def forward(self, x):
        if not self.transformer.is_identity:
            beta = self.transformer.predict_parameters(x)
            mus = self.mus.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            mus = self.transformer.apply_parameters(mus, beta).permute(1, 0, 2, 3, 4)
            sigmas = self.sigmas.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            sigmas = self.transformer.apply_parameters(sigmas, beta, is_var=True).permute(1, 0, 2, 3, 4)
        else:
            mus, sigmas = self.mus, self.sigmas

        x = x.unsqueeze(0).expand(self.n_prototypes, -1, -1, -1, -1)
        log_mixing_probs = torch.log_softmax(self.mixing_params, 0)
        log_probs = torch.stack([self.log_pdf(x_k, m, s) for x_k, m, s in zip(x, mus, sigmas)], 1)
        log_weighted_probs = log_probs + log_mixing_probs
        with torch.no_grad():
            gamma = torch.exp(log_weighted_probs - torch.logsumexp(log_weighted_probs, 1).unsqueeze(1))
        return -(gamma * log_weighted_probs).sum(1).mean(), -log_probs

    def log_pdf(self, x, mu, sigma, weights=None):
        prec = 1 / (sigma**2 + self.var_min)
        gaussian_sq_error = (x**2*prec - 2*x*mu*prec + mu**2*prec - torch.log(prec)).flatten(1)
        if self.loss_weights is not None:
            gaussian_sq_error = gaussian_sq_error * self.loss_weights.flatten()
        return -0.5 * (gaussian_sq_error.sum(1) + self.n * torch.log(self.const))

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        else:
            mus = self.mus.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            return self.transformer(x, mus)[1]

    def step(self):
        self.transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        self.transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)
            elif name == 'prototypes':
                if isinstance(param, nn.Parameter):
                    param = param.data
                state['mus'].copy_(param)
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
        self.mus[i].data.copy_(copy_with_noise(self.mus[j], NOISE_SCALE))
        self.sigmas[i].data.copy_(copy_with_noise(self.sigmas[j], 0))
        self.mixing_params[i].data.copy_(self.mixing_params[j].clone())
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(self, 'optimizer'):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                for param in [self.mus, self.sigmas, self.mixing_params]:
                    opt.state[param]['exp_avg'][i] = opt.state[param]['exp_avg'][j]
                    opt.state[param]['exp_avg_sq'][i] = opt.state[param]['exp_avg_sq'][j]
            else:
                raise NotImplementedError('unknown optimizer: you should define how to reinstanciate statistics if any')
