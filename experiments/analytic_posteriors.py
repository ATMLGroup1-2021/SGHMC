import pyro
import pyro.distributions as dist
import torch
from torch.utils.data import Dataset
from functools import partial
from torch import matmul


class _2dGaussianDataset(Dataset):
    def __init__(self, mean, covar, size):
        d = dist.MultivariateNormal(mean, covariance_matrix=covar)
        self.size = size
        self.dataset = d.sample((size,))
        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[index]


def _2d_gaussian_model(data, likelihood_covar):
    mean = pyro.sample('mean', dist.MultivariateNormal(torch.zeros(2),
                                                       covariance_matrix=torch.diag(torch.tensor([5, 5]))))
    obs = pyro.sample('obs', dist.MultivariateNormal(mean, likelihood_covar), obs=data)
    return obs


def _2d_gaussian(likelihood_mean, likelihood_covar, num_samples):
    '''
    Takes in two (real) tensors, mean (2x1) and covar (2x2), creates dataset
    Returns model, dataset, and analytic posterior
    '''

    dataset = _2dGaussianDataset(likelihood_mean, likelihood_covar, num_samples)
    model = partial(_2d_gaussian_model, likelihood_covar)

    prior_mean = torch.zeros(2)
    prior_covar = torch.diag(torch.tensor([5., 5.]))
    inv_prior_covar = torch.inverse(prior_covar)
    inv_likelihood_covar = torch.inverse(likelihood_covar)

    sample_mean = dataset.dataset.mean(dim=0)

    posterior_covar = torch.inverse(inv_prior_covar + num_samples * inv_likelihood_covar)
    posterior_mean = matmul(posterior_covar,
                            matmul(inv_prior_covar, prior_mean) +
                            num_samples * matmul(inv_likelihood_covar, sample_mean))

    return dataset, model, dist.MultivariateNormal(posterior_mean, covariance_matrix=posterior_covar)
