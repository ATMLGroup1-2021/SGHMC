import torch 
import matplotlib.pyplot as plt 

from pyro.infer.mcmc import MCMC 
from pyro.infer.mcmc.nuts import HMC

import distributions

# Define HMC sampler - using pyro inbuilt HMC 
def hmc_sampling(dist):
    hmc_kernel = HMC(dist, step_size=0.9, num_steps=4)
    mcmc_sampler = MCMC(hmc_kernel, 
                    num_samples = 1000,
                    warmup_steps = 50)
    mcmc_sampler.run() # pass dataloader here for sghmc 
    mcmc_sampler.summary()
    posterior = mcmc_sampler.get_samples()
    return posterior


# Unimodal distributions
fig, ax1 = plt.subplots(1, 2, figsize=(16, 4))

unimodal_sym_posterior = hmc_sampling(distributions.unimodal_sym)
ax1[0].hist(unimodal_sym_posterior['weight'].numpy(), rwidth = 0.3)
ax1[0].set_title('Unimodal symmetric distribution with inbuilt HMC sampler')

unimodal_unsym_posterior = hmc_sampling(distributions.unimodal_unsym)
ax1[1].hist(unimodal_unsym_posterior['weight'].numpy(), rwidth= 0.3)
ax1[1].set_title('Unimodal asymmetric distribution with inbuilt HMC sampler')

plt.show()

# Bimodal distributions 
fig, ax2 = plt.subplots(1, 2, figsize=(16, 4))

bimodal_symmetric_posterior = hmc_sampling(distributions.bimodal_sym)
ax2[0].hist(bimodal_symmetric_posterior['a'].numpy(), rwidth = 0.3) # TODO - plot the correct key, 'weight' when distribution is fixed
ax2[0].set_title('Bimodal symmetric distribution with inbuilt HMC sampler')

bimodal_symmetric_posterior = hmc_sampling(distributions.bimodal_unsym)
ax2[1].hist(bimodal_symmetric_posterior['a'].numpy(), rwidth = 0.3) # TODO - same as above, also for trimodals
ax2[1].set_title('Bimodal asymmetric distribution with inbuilt HMC sampler')
plt.show()

# Trimodal distributions
fig, ax3 = plt.subplots(1, 2, figsize=(16, 4))

trimodal_symmetric_posterior = hmc_sampling(distributions.trimodal_sym)
ax3[0].hist(trimodal_symmetric_posterior['a'].numpy(), rwidth = 0.3)
ax3[0].set_title('Trimodal symmetric distribution with inbuilt HMC sampler')

trimodal_symmetric_posterior = hmc_sampling(distributions.trimodal_unsym)
ax3[1].hist(trimodal_symmetric_posterior['a'].numpy(), rwidth = 0.3)
ax3[1].set_title('Trimodal asymmetric distribution with inbuilt HMC sampler')
plt.show()