import torch 
import matplotlib.pyplot as plt 

from pyro.infer.mcmc import MCMC 
from pyro.infer.mcmc.nuts import HMC

import distributions

# Define HMC kernel
def hmc_sampling(dist):
    hmc_kernel = HMC(dist, step_size=0.9, num_steps=4)
    mcmc_sampler = MCMC(hmc_kernel, 
                    num_samples = 1000,
                    warmup_steps = 50)
    mcmc_sampler.run()
    mcmc_sampler.summary()
    posterior = mcmc_sampler.get_samples()
    return posterior

posteriors = []
posteriors.append(hmc_sampling(distributions.unimodal_sym))
posteriors.append(hmc_sampling(distributions.unimodal_unsym))

plt.figure(figsize=(10, 10))
for index, posterior in enumerate(posteriors):
    plt.subplot(10, 4, index + 1)
    plt.hist(posterior['weight'].numpy())
plt.show()
