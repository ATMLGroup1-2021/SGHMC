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
import plotly.graph_objects as go
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

'''
Perform testing for SGHMC and HMC using conjugate prior
distributions, inorder to compare with analytical posterior
'''


def calculate_sghmc(dataset, model, step_size=0.1, num_steps=4, friction=0.1, batch_size=256, num_samples=1000,
                    num_burnin=200, resample_r_freq=1):
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    sghmc_kernel = SGHMC(model, step_size=step_size, num_steps=num_steps, friction=friction,
                         resample_r_freq=resample_r_freq)
    mcmc = MCMC(sghmc_kernel, num_samples=num_samples, warmup_steps=num_burnin)
    mcmc.run(data_loader)

    return mcmc.get_samples()


def calculate_hmc(dataset, model, step_size=0.1, num_steps=4, num_samples=1000, num_burnin=200):
    hmc_kernel = BasicHMC(model, step_size=step_size, num_steps=num_steps)
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=num_burnin)
    mcmc.run(dataset.dataset)

    return mcmc.get_samples()


def _2d_gauss_sghmc(likelihood_mean, likelihood_var, dataset_size, step_size=0.1, num_steps=4, friction=0.1,
                    batch_size=256, num_samples=1000, num_burnin=200, resample_r_freq=1):
    # posterior is over likelihood mean
    dataset, model, posterior = _2d_gaussian(likelihood_mean, likelihood_var, dataset_size)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size, num_steps=num_steps, friction=friction,
                                    batch_size=batch_size, num_samples=num_samples, num_burnin=num_burnin,
                                    resample_r_freq=resample_r_freq)['mean']
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size, num_steps=num_steps, num_samples=num_samples,
                                num_burnin=num_burnin)['mean']

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
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size)['rate']

    return sghmc_samples, hmc_samples, posterior


def _2d_scatter_plot(sghmc_samples, hmc_samples, analytical_posterior):
    sghmc_plot = go.Scatter(
        x=sghmc_samples[:, 0].tolist()[0::5],
        y=sghmc_samples[:, 1].tolist()[0::5],
        mode='markers',
        marker=dict(color="darkturquoise"),
    )

    hmc_plot = go.Scatter(
        x=hmc_samples[:, 0].tolist()[0::10],
        y=hmc_samples[:, 1].tolist()[0::10],
        mode='markers',
        marker=dict(color="hotpink")
    )

    x_mean = analytical_posterior.mean[0].item()
    y_mean = analytical_posterior.mean[1].item()
    xrange = torch.linspace(x_mean - 1, x_mean + 1, 100)
    yrange = torch.linspace(y_mean - 1, y_mean + 1, 100)
    x_coords, y_coords = torch.meshgrid(xrange, yrange)
    coords = torch.stack((x_coords, y_coords), dim=2)

    max_analytical = analytical_posterior.log_prob(analytical_posterior.mean).exp().item()
    dist_plot = analytical_posterior.log_prob(coords).exp()

    analytic_plot = go.Contour(
        z=dist_plot,
        x=xrange.tolist(),  # horizontal axis
        y=yrange.tolist(),  # vertical axis
        contours_coloring='lines',
        contours=dict(
            start=0.1,
            end=max_analytical,
            size=0.1*max_analytical,
        )
    )

    fig = go.Figure(data=(analytic_plot, sghmc_plot, hmc_plot))
    fig.show()


def _2d_trajectory_plot(samples, analytical_posterior):
    hmc_plot = go.Scatter(
        x=samples[:, 0].tolist()[-50::],
        y=samples[:, 1].tolist()[-50::],
        mode='lines+markers',
        marker=dict(color="darkturquoise"),
        line=dict(width=1, color="darkturquoise"),
    )

    x_mean = analytical_posterior.mean[0].item()
    y_mean = analytical_posterior.mean[1].item()
    xrange = torch.linspace(x_mean - 1, x_mean + 1, 100)
    yrange = torch.linspace(y_mean - 1, y_mean + 1, 100)
    x_coords, y_coords = torch.meshgrid(xrange, yrange)
    coords = torch.stack((x_coords, y_coords), dim=2)

    max_analytical = analytical_posterior.log_prob(analytical_posterior.mean).exp().item()
    dist_plot = analytical_posterior.log_prob(coords).exp()

    analytic_plot = go.Contour(
        z=dist_plot,
        x=xrange.tolist(),  # horizontal axis
        y=yrange.tolist(),  # vertical axis
        contours_coloring='lines',
        contours=dict(
            start=0.1,
            end=max_analytical,
            size=0.1 * max_analytical,
        )
    )

    fig = go.Figure(data=(analytic_plot, hmc_plot))
    fig.show()


def _2d_contour_plot(sghmc_samples, hmc_samples, analytical_posterior):
    sghmc_bins, xedges, yedges = np.histogram2d(sghmc_samples[:, 0].tolist(), sghmc_samples[:, 1].tolist(),
                                                bins=[10, 10])
    xbins = dict(size=xedges[1]-xedges[0],)
    ybins = dict(size=yedges[1] - yedges[0],)
    max_sghmc = max(map(max, sghmc_bins))

    sghmc_plot = go.Histogram2dContour(
        x=sghmc_samples[:, 0].tolist(),
        y=sghmc_samples[:, 1].tolist(),
        contours_coloring="lines",
        colorscale=[[0, "blue"], [1, "blue"]],
        contours=dict(
            start=0.1 * max_sghmc,
            end=max_sghmc,
            size=0.1 * max_sghmc,
        ),
        xbins=xbins,
        ybins=ybins,
    )

    hmc_bins, xedges, yedges = np.histogram2d(hmc_samples[:, 0].tolist(), hmc_samples[:, 1].tolist(), bins=[10, 10])
    xbins = dict(size=xedges[1] - xedges[0], )
    ybins = dict(size=yedges[1] - yedges[0], )
    max_hmc = max(map(max, sghmc_bins))

    hmc_plot = go.Histogram2dContour(
        x=hmc_samples[:, 0].tolist(),
        y=hmc_samples[:, 1].tolist(),
        contours_coloring="lines",
        colorscale=[[0, "green"], [1, "green"]],
        contours=dict(
            start=0.1 * max_hmc,
            end=max_hmc,
            size=0.1 * max_hmc,
        ),
        xbins=xbins,
        ybins=ybins,
    )

    x_mean = analytical_posterior.mean[0].item()
    y_mean = analytical_posterior.mean[1].item()
    xrange = torch.linspace(x_mean - 1, x_mean + 1, 100)
    yrange = torch.linspace(y_mean - 1, y_mean + 1, 100)
    x_coords, y_coords = torch.meshgrid(xrange, yrange)
    coords = torch.stack((x_coords, y_coords), dim=2)

    max_analytical = analytical_posterior.log_prob(analytical_posterior.mean).exp().item()
    dist_plot = analytical_posterior.log_prob(coords).exp()

    analytic_plot = go.Contour(
        z=dist_plot,
        x=xrange.tolist(),  # horizontal axis
        y=yrange.tolist(),  # vertical axis
        contours_coloring='lines',
        colorscale=[[0, "red"], [1, "red"]],
        contours=dict(
            start=0.1 * max_analytical,
            end=max_analytical,
            size=0.1*max_analytical,
        ),
    )

    fig = go.Figure(data=(analytic_plot, hmc_plot, sghmc_plot))
    fig.show()


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
    gauss_sghmc, gauss_hmc, analytic_posterior = \
        _2d_gauss_sghmc(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 2000, batch_size=512,
                        step_size=0.01, num_steps=4, num_samples=1000, num_burnin=200, resample_r_freq=25, friction=10)

    # _2d_scatter_plot(gauss_sghmc, gauss_hmc, analytic_posterior)
    # _2d_trajectory_plot(gauss_sghmc, analytic_posterior)
    # _2d_trajectory_plot(gauss_hmc, analytic_posterior)
    _2d_contour_plot(gauss_sghmc, gauss_hmc, analytic_posterior)

    # _1d_plot(*(beta_bernoulli_sghmc(0.7, 1000, step_size=0.005)))
    #
    # _1d_plot(*(gamma_poisson_sghmc(10, 1000, step_size=0.005)))


if __name__ == "__main__":
    main()