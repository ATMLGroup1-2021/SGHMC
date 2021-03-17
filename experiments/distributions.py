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


class CustomDist(pyro.distributions.Uniform):
    def __init__(self, pdf, min_v, max_v):
        super().__init__(min_v, max_v, None)
        self.pdf = pdf

    def log_prob(self, value):
        return torch.log(self.pdf(value))


# Denumerator obtained by integrating the pdf from -2 to 2 in wolframalpha
def fig1_pdf(value):
    p = torch.exp(2 * value ** 2 - value ** 4) / 5.365134
    p[(value < -2) | (value > 2)] = 0
    return p


def fig1_model(batch):
    x = pyro.sample("x", CustomDist(fig1_pdf, -2, 2))
    return x


fig3_dist = dist.MultivariateNormal(torch.tensor([0., 0.]), covariance_matrix=torch.tensor([[1., 0.9], [0.9, 1.]]))

def fig3_model(batch):
    x = pyro.sample("x", fig3_dist)
    return x

# to see the distribution
# plt.hist([bimodal_sym() for _ in range(10000)], bins="auto")
# plt.xlabel('value')
# plt.ylabel('count')
# plt.show()
