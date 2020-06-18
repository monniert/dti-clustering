import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn.cluster import KMeans

from .tools import copy_with_noise, generate_data

SIGMA_MIN = 0.25
SIGMA_PRIOR = 0.5


class GaussianMixtureModel(nn.Module):
    def __init__(self, dataset, n_clusters, **kwargs):
        super().__init__()
        self.n_clusters = n_clusters
        init_type = kwargs.get('init_type', 'kmeans')
        sigma = kwargs.get('sigma_prior', SIGMA_PRIOR)
        sigma_min = kwargs.get('sigma_min', SIGMA_MIN)
        if init_type == 'kmeans':
            mus, sigmas = self._kmeans_init(dataset, n_clusters, sigma_min)
        else:
            mus = generate_data(dataset, K=n_clusters, init_type=init_type)
            sigmas = generate_data(dataset, K=n_clusters, init_type='constant', value=sigma)

        self.mus = nn.ParameterList(list(map(nn.Parameter, mus)))
        self.sigmas = nn.ParameterList(list(map(nn.Parameter, sigmas)))
        self.sigma_min = sigma_min
        self.var_min = sigma_min**2
        self.n = mus[0].numel()
        self.register_buffer('const', torch.Tensor([2*np.pi]))
        self.mixing_params = nn.Parameter(torch.ones(n_clusters))

    @staticmethod
    def _kmeans_init(dataset, n_clusters, sigma_min=0.1):
        images = next(iter(DataLoader(dataset, batch_size=100*n_clusters, shuffle=True, num_workers=4)))[0]
        img_size = images.shape[1:]
        X = images.flatten(1).numpy()
        cluster = KMeans(n_clusters)
        labels = cluster.fit_predict(X)
        mus = list(map(lambda c: torch.Tensor(c).reshape(img_size), cluster.cluster_centers_))
        sigmas = []
        for k in range(n_clusters):
            s = np.sqrt(np.mean((X[labels == k] - cluster.cluster_centers_[k])**2, axis=0))
            sigmas.append(torch.Tensor(s).reshape(img_size).clamp(sigma_min))
        return mus, sigmas

    def forward(self, x, transformer=None):
        if transformer is not None and not transformer.is_identity:
            mus = [m.unsqueeze(0).expand(x.size(0), -1, -1, -1) for m in self.mus]
            sigmas = [s.unsqueeze(0).expand(x.size(0), -1, -1, -1) for s in self.sigmas]
            beta = transformer.predict_parameters(x)
            sigmas = transformer.apply_parameters(x, sigmas, beta, is_var=True)[1]
            x, mus = transformer.apply_parameters(x, mus, beta)
            sigmas = sigmas.permute(1, 0, 2, 3, 4)
            mus = mus.permute(1, 0, 2, 3, 4)
        else:
            x = x.unsqueeze(1).expand(-1, self.n_clusters, -1, -1, -1)
            mus, sigmas = self.mus, self.sigmas

        x = x.permute(1, 0, 2, 3, 4)  # KBCHW
        log_mixing_probs = torch.log_softmax(self.mixing_params, 0)
        log_probs = torch.stack([self.log_pdf(x_k, m, s) for x_k, m, s in zip(x, mus, sigmas)], 1)
        log_weighted_probs = log_probs + log_mixing_probs
        with torch.no_grad():
            gamma = torch.exp(log_weighted_probs - torch.logsumexp(log_weighted_probs, 1).unsqueeze(1))
        return -(gamma * log_weighted_probs).sum(1).mean(), -log_probs

    def log_pdf(self, x, mu, sigma):
        var = sigma**2 + self.var_min
        gaussian_sq_error = ((x - mu)**2 / var + torch.log(var)).flatten(1)
        return -0.5 * (gaussian_sq_error.sum(1) + self.n * torch.log(self.const))

    def restart_branch_from(self, i, j, noise_scale=0.001):
        self.mus[i].data.copy_(copy_with_noise(self.mus[j], noise_scale))
        self.sigmas[i].data.copy_(copy_with_noise(self.sigmas[j], 0))
        self.mixing_params.data[i] = self.mixing_params.data[j]
