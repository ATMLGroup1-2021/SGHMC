import matplotlib.pyplot as plt
import torch
from pyro.infer import MCMC
from torch.utils.data import Dataset, DataLoader

from experiments.distributions import fig1_model, CustomDist, fig1_pdf, fig3_model, fig3_dist
from samplers.basic_hmc import BasicHMC
from samplers.sghmc import SGHMC
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, ):
        super().__init__()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor([0.0])

data_loader = DataLoader(DummyDataset(), 1, shuffle=True)


def fig1_experiment():
    model1 = fig1_model

    x = torch.linspace(-2, 2, 128)
    dist_plot = CustomDist(fig1_pdf, -2, 2).log_prob(x).exp()

    sghmc_kernel = SGHMC(model1, step_size=0.1, num_steps=4)
    mcmc = MCMC(sghmc_kernel, num_samples=2000, warmup_steps=500)
    mcmc.run(data_loader)

    samples_sghmc = mcmc.get_samples()['x']
    counts, bin_edges = np.histogram(samples_sghmc, bins=50, range=[-2, 2], normed=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    sghmc_dist = np.stack([bin_centres, counts])
    plt.plot(sghmc_dist[0], sghmc_dist[1])

    hmc_kernel = BasicHMC(model1, step_size=0.1, num_steps=4)
    mcmc = MCMC(hmc_kernel, num_samples=2000, warmup_steps=500, num_chains=1)
    mcmc.run(DummyDataset().__getitem__(0))

    samples_hmc = mcmc.get_samples()['x']
    counts, bin_edges = np.histogram(samples_hmc, bins=50, range=[-2, 2], normed=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hmc_dist = np.stack([bin_centres, counts])

    plt.plot(x, dist_plot)
    plt.plot(sghmc_dist[0], sghmc_dist[1])
    plt.plot(hmc_dist[0], hmc_dist[1])
    plt.show()


def fig3_experiment():
    model3 = fig3_model
    x = torch.linspace(-2, 3, 256)
    y = torch.linspace(-2, 3, 256)
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack((xx, yy), dim=-1)
    dist_plot = fig3_dist.log_prob(xy.reshape(-1, 2)).exp().reshape(256, 256)

    sghmc_kernel = SGHMC(model3, step_size=0.1, num_steps=4)
    mcmc = MCMC(sghmc_kernel, num_samples=2000, warmup_steps=500)
    mcmc.run(data_loader)
    samples_sghmc = mcmc.get_samples(50)['x']

    plt.figure(figsize=(6, 6))
    plt.contour(x, y, dist_plot)
    plt.scatter(samples_sghmc[:, 0], samples_sghmc[:, 1], facecolors='none', edgecolors='r')
    plt.show()


if __name__ == "__main__":
    fig1_experiment()
    fig3_experiment()
