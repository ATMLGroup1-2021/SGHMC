import torch
from pyro.infer import MCMC
from pyro.infer.mcmc import HMC
import pyro.distributions as dist
import pyro
from torch.utils.data import Dataset, DataLoader
import itertools

from sghmc import SGHMC

true_coefs = torch.tensor([1., 2., 3.])
data = torch.randn(2000, 3)
dim = 3
labels = dist.Binomial(logits=(true_coefs * data).sum(-1)).sample()

class TmpData(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        data = torch.randn(3)
        label = dist.Binomial(logits=(true_coefs * data).sum(-1)).sample()
        return data, label

def model(batch):
    data, label = batch
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
    y = pyro.sample('y', dist.Binomial(logits=(coefs * data).sum(-1)), obs=label)
    return y

data_loader = DataLoader(TmpData(), 32)

sghmc_kernel = SGHMC(model, step_size=0.0855, num_steps=4)
mcmc = MCMC(sghmc_kernel, num_samples=500, warmup_steps=100)
mcmc.run(data_loader)
mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP