import torch
from pyro.infer import MCMC
from pyro.infer.mcmc import HMC
import pyro.distributions as dist
import pyro

from basic_hmc import BasicHMC
from sghmc import SGHMC

def scale():
    weight = pyro.sample("weight", dist.Normal(10, 1.0))
    measurement = pyro.sample("measurement", dist.Normal(weight, 0.75))
    return measurement

conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(14.)})

sghmc_kernel = BasicHMC(conditioned_scale, step_size=0.0855, num_steps=4)
mcmc = MCMC(sghmc_kernel, num_samples=500, warmup_steps=100)
mcmc.run()
mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP