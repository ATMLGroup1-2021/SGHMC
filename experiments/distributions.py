import torch
import pyro
pyro.set_rng_seed(101)

import matplotlib.pyplot as plt
import pyro.distributions as dist

def unimodal_sym():
    return pyro.sample("weight", dist.Normal(0, 1))

def unimodal_unsym():
    return pyro.sample("weight", dist.Beta(2,5))

# not sure if bimodal and trimodal distributions are returning correctly
def bimodal_sym():
    a = pyro.sample("a", dist.Normal(-1, 0.5))
    b = pyro.sample("b", dist.Normal(1, 0.5))
    w = pyro.sample("weight", dist.Bernoulli(0.5))
    return (a,b), w

def bimodal_unsym():
    a = pyro.sample("a", dist.Normal(3, 2))
    b = pyro.sample("b", dist.Normal(10, 1))
    w = pyro.sample("weight", dist.Bernoulli(0.5))
    return (a,b), w

def trimodal_sym():
    a = pyro.sample("a", dist.Normal(-2, 0.5))
    b = pyro.sample("b", dist.Normal(0, 0.5))
    c = pyro.sample("c", dist.Normal(2, 0.5))
    w = pyro.sample("weight", dist.Categorical(torch.tensor([1/3,1/3,1/3])))
    return (a,b,c), w

def trimodal_unsym():
    a = pyro.sample("a", dist.Normal(-4, 2))
    b = pyro.sample("b", dist.Normal(1, 1))
    c = pyro.sample("c", dist.Normal(8, 0.6))
    w = pyro.sample("w", dist.Categorical(torch.tensor([1/3,1/3,1/3])))
    return (a,b,c), w

# to see the distribution
# plt.hist([bimodal_sym() for _ in range(10000)], bins="auto")
# plt.xlabel('value')
# plt.ylabel('count')
# plt.show()
