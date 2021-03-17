import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, HMC
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from basic_hmc import BasicHMC
from sghmc import SGHMC


class TmpData(Dataset):
    def __init__(self, fail, succ):
        self.fail = fail
        self.succ = succ
        super().__init__()

    def __len__(self):
        return self.fail + self.succ

    def __getitem__(self, index):
        if index < self.fail:
            return torch.tensor([0.0])
        else:
            return torch.tensor([1.0])


def model(label):
    p = pyro.sample('p', dist.Beta(1, 1))
    y = pyro.sample('y', dist.Binomial(probs=p), obs=label)
    return y

fail = 256
succ = 64
n = fail + succ

data_loader = DataLoader(TmpData(fail, succ), 64, shuffle=True)
d = TmpData(fail, succ)
data = torch.tensor([d.__getitem__(i) for i in range(n)])

sghmc_kernel = SGHMC(model, step_size=0.025, num_steps=16)
mcmc = MCMC(sghmc_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data_loader)

samples_sghmc = mcmc.get_samples()['p']

sghmc_kernel = BasicHMC(model, step_size=0.025, num_steps=16)
mcmc = MCMC(sghmc_kernel, num_samples=1000, warmup_steps=200, num_chains=1)
mcmc.run(data)
samples_hmc = mcmc.get_samples()['p']


analytical_posterior = dist.Beta(1 + succ, 1 + fail)

x = torch.linspace(0, 1, 128)

dist_plot = analytical_posterior.log_prob(x).exp()
plt.plot(x, dist_plot); plt.show()

plt.hist(samples_sghmc.tolist(), 100, range=[0.0, 1.0]); plt.show()

plt.hist(samples_hmc.tolist(), 100, range=[0.0, 1.0]); plt.show()