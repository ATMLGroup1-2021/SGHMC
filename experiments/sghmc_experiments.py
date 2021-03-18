import os
import sys
import matplotlib.pyplot as plt
from samplers.sghmc import SGHMC
import numpy as np
from samplers.basic_hmc import BasicHMC
from analytic_posteriors import beta_bernoulli, _2d_gaussian, gamma_poisson
import torch
from torch.utils.data import DataLoader
from pyro.infer import MCMC
import plotly.express as px
import plotly.graph_objects as go
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

'''
Perform testing for SGHMC and HMC using conjugate prior
distributions, inorder to compare with analytical posterior
'''


def calculate_sghmc(dataset, model, step_size=0.1):
    data_loader = DataLoader(dataset, 256, shuffle=True)

    sghmc_kernel = SGHMC(model, step_size=step_size, num_steps=4)
    mcmc = MCMC(sghmc_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(data_loader)

    return mcmc.get_samples()


def calculate_hmc(dataset, model, step_size=0.1):
    hmc_kernel = BasicHMC(model, step_size=step_size, num_steps=4)
    mcmc = MCMC(hmc_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(dataset.dataset)

    return mcmc.get_samples()


def _2d_gauss_sghmc(likelihood_mean, likelihood_var, num_samples, step_size=0.1):
    # posterior is over likelihood mean
    dataset, model, posterior = _2d_gaussian(likelihood_mean, likelihood_var, num_samples)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size)['mean']
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size)['mean']

    return sghmc_samples, hmc_samples, posterior


def beta_bernoulli_sghmc(p, num_samples, step_size=0.1):
    # posterior is over p
    dataset, model, posterior = beta_bernoulli(p, num_samples)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size)['p']
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size)['p']

    return sghmc_samples, hmc_samples, posterior


def gamma_poisson_sghmc(rate, num_samples, step_size=0.1):
    # posterior is over rate
    dataset, model, posterior = gamma_poisson(rate, num_samples)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size)['rate']
    hmc_samples = calculate_hmc(dataset, model, step_size=0.005)['rate']

    return sghmc_samples, hmc_samples, posterior


def _2d_plot(sghmc_samples, hmc_samples, analytical_posterior):
    # fig = px.density_contour(x=samples[:, 0].tolist(), y=samples[:, 1].tolist())
    # fig.show()
    # plt.hist2d(samples[:, 0].tolist(), samples[:, 1].tolist(), bins=[20, 20])
    # plt.show()

    x_mean = analytical_posterior.mean[0].item()
    y_mean = analytical_posterior.mean[1].item()
    x_coords, y_coords = torch.meshgrid(torch.linspace(x_mean - 2, x_mean + 2, 100),
                                        torch.linspace(y_mean - 2, y_mean + 2, 100))
    coords = torch.stack((x_coords, y_coords), dim=2)

    max = analytical_posterior.log_prob(analytical_posterior.mean).exp().item()
    dist_plot = analytical_posterior.log_prob(coords).exp()

    sghmc_plot = go.Histogram2dContour(
        x=sghmc_samples[:, 0].tolist(),
        y=sghmc_samples[:, 1].tolist(),
        contours_coloring="lines",
        colorscale=[[0, "blue"], [1, "blue"]],
        # contours=dict(
        #     start=0.1,
        #     end=1,
        #     size=0.1,
        # ),
        # histnorm="probability"
    )

    hmc_plot = go.Histogram2dContour(
        x=hmc_samples[:, 0].tolist(),
        y=hmc_samples[:, 1].tolist(),
        contours_coloring="lines",
        colorscale=[[0, "green"], [1, "green"]],
        # contours=dict(
        #     start=0.1,
        #     end=1,
        #     size=0.1,
        # ),
        # histnorm="probability"
    )

    analytic_plot = go.Contour(
        z=dist_plot,
        x=torch.linspace(x_mean - 2, x_mean + 2, 100).tolist(),  # horizontal axis
        y=torch.linspace(y_mean - 2, y_mean + 2, 100).tolist(),  # vertical axis
        contours_coloring='lines',
        colorscale=[[0, "red"], [1, "red"]],
        contours=dict(
            start=0.1*max,
            end=max,
            size=0.1*max,
        ),
    )

    fig = go.Figure(data=(analytic_plot, hmc_plot, sghmc_plot))
    fig.show()
    # plt.contour(x_coords, y_coords, dist_plot)
    # plt.show()


def _1d_plot(sghmc_samples, hmc_samples, analytical_posterior):
    counts, bin_edges = np.histogram(sghmc_samples, bins=25, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    sghmc_dist = np.stack([bin_centres, counts])
    l1, = plt.plot(sghmc_dist[0], sghmc_dist[1], c='r')

    counts, bin_edges = np.histogram(hmc_samples, bins=25, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hmc_dist = np.stack([bin_centres, counts])
    l2, = plt.plot(hmc_dist[0], hmc_dist[1], c='g')

    mean = analytical_posterior.mean
    max = analytical_posterior.log_prob(mean).exp().item()
    coords = torch.linspace(mean.item()-5*(1/max), mean.item()+5*(1/max), 256)
    dist_plot = analytical_posterior.log_prob(coords).exp()
    l0, = plt.plot(coords, dist_plot, c='b')

    plt.ylabel("Probability density")
    plt.xlabel(r"$\theta$")
    plt.legend([l0, l1, l2], ["PDF", "SGHMC", "HMC"])
    plt.show()


def main():
    # _2d_plot(*(_2d_gauss_sghmc(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 1000)))

    _1d_plot(*(beta_bernoulli_sghmc(0.7, 1000, step_size=0.05)))

    _1d_plot(*(gamma_poisson_sghmc(10, 1000, step_size=0.005)))


if __name__ == "__main__":
    main()