import os, sys, torch
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC

import distributions

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.basic_hmc import BasicHMC
from samplers.sghmc import SGHMC

# usage:
# python sampling_experiments.py builtin
# python sampling_experiments.py basic
# python sampling_experiments.py sghmc

# result will be saved in figures/sampling_experiments

kernel = None
option = sys.argv[1]
if sys.argv[1] == "basic": kernel = BasicHMC
elif sys.argv[1] == "builtin": kernel = HMC
else: kernel = SGHMC

# Define HMC sampler - using pyro inbuilt HMC
def hmc_sampling(dist):
    hmc_kernel = kernel(dist, step_size=0.9, num_steps=4)
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
ax1[0].hist(unimodal_sym_posterior['sample'].numpy(), rwidth = 0.3)
ax1[0].set_title('Unimodal symmetric distribution with ' + option + ' HMC sampler')

unimodal_asym_posterior = hmc_sampling(distributions.unimodal_asym)
ax1[1].hist(unimodal_asym_posterior['sample'].numpy(), rwidth= 0.3)
ax1[1].set_title('Unimodal asymmetric distribution with ' + option + ' HMC sampler')
plt.savefig("figures/sampling_experiments/unimodal_" + option + ".jpg")

# Bimodal distributions
fig, ax2 = plt.subplots(1, 2, figsize=(16, 4))

bimodal_symmetric_posterior = hmc_sampling(distributions.bimodal_sym)
sample1, sample2, weight = bimodal_symmetric_posterior["sample1"],bimodal_symmetric_posterior["sample2"],bimodal_symmetric_posterior["weight"]
bimodal_symmetric_posterior = np.array([[sample1[i],sample2[i]][weight[i]<0.5] for i in range(len(sample1))])

ax2[0].hist(bimodal_symmetric_posterior, rwidth = 0.3)
ax2[0].set_title('Bimodal symmetric distribution with ' + option + ' HMC sampler')

bimodal_asymmetric_posterior = hmc_sampling(distributions.bimodal_asym)
sample1, sample2, weight = bimodal_asymmetric_posterior["sample1"],bimodal_asymmetric_posterior["sample2"],bimodal_asymmetric_posterior["weight"]
bimodal_asymmetric_posterior = np.array([[sample1[i],sample2[i]][weight[i]<0.5] for i in range(len(sample1))])

ax2[1].hist(bimodal_asymmetric_posterior, rwidth = 0.3)
ax2[1].set_title('Bimodal asymmetric distribution with ' + option + ' HMC sampler')
plt.savefig("figures/sampling_experiments/bimodal_" + option + ".jpg")

# Trimodal distributions
fig, ax3 = plt.subplots(1, 2, figsize=(16, 4))

trimodal_symmetric_posterior = hmc_sampling(distributions.trimodal_sym)
sample1, sample2, sample3, weight = trimodal_symmetric_posterior["sample1"],trimodal_symmetric_posterior["sample2"],trimodal_symmetric_posterior["sample3"],trimodal_symmetric_posterior["weight"]
trimodal_symmetric_posterior = np.array([[sample1[i],sample2[i],sample3[i]][0 if weight[i]<1/3 else (2 if weight[i]>2/3 else 1)] for i in range(len(sample1))])

ax3[0].hist(trimodal_symmetric_posterior, rwidth = 0.3)
ax3[0].set_title('Trimodal symmetric distribution with ' + option + ' HMC sampler')

trimodal_asymmetric_posterior = hmc_sampling(distributions.trimodal_asym)
sample1, sample2, sample3, weight = trimodal_asymmetric_posterior["sample1"],trimodal_asymmetric_posterior["sample2"],trimodal_asymmetric_posterior["sample3"],trimodal_asymmetric_posterior["weight"]
trimodal_asymmetric_posterior = np.array([[sample1[i],sample2[i],sample3[i]][0 if weight[i]<1/3 else (2 if weight[i]>2/3 else 1)] for i in range(len(sample1))])

ax3[1].hist(trimodal_asymmetric_posterior, rwidth = 0.3)
ax3[1].set_title('Trimodal asymmetric distribution with ' + option + ' HMC sampler')
plt.savefig("figures/sampling_experiments/trimodal_" + option + ".jpg")
