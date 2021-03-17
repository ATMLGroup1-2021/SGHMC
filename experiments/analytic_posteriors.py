import pyro
import pyro.distributions as dist
import torch
from torch.utils.data import Dataset
from functools import partial
from torch import matmul


class DistributionDataset(Dataset):
    def __init__(self, dist, num_samples):
        self.num_samples = num_samples
        self.dataset = dist.sample((num_samples,))
        super().__init__()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.dataset[index]


def _2d_gaussian_model(data, likelihood_covar):
    mean = pyro.sample('mean', dist.MultivariateNormal(torch.zeros(2),
                                                       covariance_matrix=torch.diag(torch.tensor([20., 20.]))))
    obs = pyro.sample('obs', dist.MultivariateNormal(mean, likelihood_covar), obs=data)
    return obs


def _2d_gaussian(likelihood_mean, likelihood_covar, num_samples):
    '''
        Takes in two (real) tensors, mean (2x1) and covar (2x2), and num_samples, and creates dataset
        Returns model, dataset, and analytic posterior
    '''

    dataset = DistributionDataset(dist.MultivariateNormal(likelihood_mean, covariance_matrix=likelihood_covar),
                                 num_samples)
    model = partial(_2d_gaussian_model, likelihood_covar=likelihood_covar)

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


def beta_bernoulli_model(data):
    p = pyro.sample('p', dist.Beta(1, 1))
    y = pyro.sample('y', dist.Bernoulli(p), obs=data)
    return y


def beta_bernoulli(p, num_samples):
    '''
        Takes in real p and num_samples, and creates dataset of num_samples draws from Bernoulli(p)
        Returns model, dataset, and analytic posterior
    '''

    dataset = DistributionDataset(dist.Bernoulli(p), num_samples)
    model = beta_bernoulli_model

    successes = (dataset.dataset == torch.ones(num_samples)).sum().item()

    return dataset, model, dist.Beta(1 + successes, 1 + num_samples - successes)


def gamma_poisson_model(data):
    rate = pyro.sample('p', dist.Gamma(0.001, 0.001))
    y = pyro.sample('y', dist.Poisson(rate), obs=data)
    return y


def gamma_poisson(rate, num_samples):
    '''
        Takes in real rate and num_samples, and creates dataset of num_samples draws from Poisson(rate)
        Returns model, dataset, and analytic posterior
    '''

    dataset = DistributionDataset(dist.Poisson(rate), num_samples)
    model = gamma_poisson_model

    occurences = dataset.dataset.sum().item()

    return dataset, model, dist.Poisson(0.001 + occurences, 0.001 + num_samples)
