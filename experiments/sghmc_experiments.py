import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from samplers.sghmc import SGHMC
from samplers.basic_hmc import BasicHMC
from analytic_posteriors import beta_bernoulli, _2d_gaussian, gamma_poisson
import torch
from torch.utils.data import DataLoader
from pyro.infer import MCMC
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

'''
Perform testing for SGHMC and HMC using conjugate prior
distributions, inorder to compare with analytical posterior
'''

# variable we are finding posterior for
likelihood_mean = torch.tensor([-3., 10.])
likelihood_var = torch.diag(torch.tensor([5., 5.]))

gauss_dataset, gauss_model, gauss_posterior = _2d_gaussian(likelihood_mean, likelihood_var, 1000)

data_loader = DataLoader(gauss_dataset, 64, shuffle=True)

sghmc_kernel = SGHMC(gauss_model, step_size=0.1, num_steps=4)
mcmc = MCMC(sghmc_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data_loader)
samples_sghmc = mcmc.get_samples()['mean']
print(samples_sghmc[:, 1])

plt.hist2d(samples_sghmc[:, 0].tolist(), samples_sghmc[:, 1].tolist(), bins=[20, 20])
plt.show()
