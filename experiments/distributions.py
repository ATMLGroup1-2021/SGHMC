import torch
import pyro
pyro.set_rng_seed(101)

import matplotlib.pyplot as plt
import pyro.distributions as dist

def unimodal_sym():
    return pyro.sample("sample", dist.Normal(0, 1))

def unimodal_asym():
    return pyro.sample("sample", dist.Beta(2,5))

# not sure if bimodal and trimodal distributions are returning correctly
def bimodal_sym():
    a = pyro.sample("sample1", dist.Normal(-1, 0.5))
    b = pyro.sample("sample2", dist.Normal(1, 0.5))
    w = pyro.sample("weight", dist.Uniform(0,1))
    return a,b,w

def bimodal_asym():
    a = pyro.sample("sample1", dist.Normal(3, 2))
    b = pyro.sample("sample2", dist.Normal(10, 1))
    w = pyro.sample("weight", dist.Uniform(0,1))
    return a,b,w

def trimodal_sym():
    a = pyro.sample("sample1", dist.Normal(-2, 0.5))
    b = pyro.sample("sample2", dist.Normal(0, 0.5))
    c = pyro.sample("sample3", dist.Normal(2, 0.5))
    w = pyro.sample("weight", dist.Uniform(0,1))
    return a,b,c,w

def trimodal_asym():
    a = pyro.sample("sample1", dist.Normal(-4, 2))
    b = pyro.sample("sample2", dist.Normal(1, 1))
    c = pyro.sample("sample3", dist.Normal(8, 0.6))
    w = pyro.sample("weight", dist.Uniform(0,1))
    return a,b,c,w

# to see the distribution
# plt.hist([bimodal_sym() for _ in range(10000)], bins="auto")
# plt.xlabel('value')
# plt.ylabel('count')
# plt.show()
