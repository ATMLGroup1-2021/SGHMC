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
from plotly.subplots import make_subplots
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


def beta_bernoulli_sghmc(p, dataset_size, step_size=0.1, num_steps=4, friction=0.1, batch_size=256, num_samples=1000,
                         num_burnin=200, resample_r_freq=1):
    # posterior is over p
    dataset, model, posterior = beta_bernoulli(p, dataset_size)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size, num_steps=num_steps, friction=friction,
                                    batch_size=batch_size, num_samples=num_samples, num_burnin=num_burnin,
                                    resample_r_freq=resample_r_freq)['p']
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size, num_steps=num_steps, num_samples=num_samples,
                                num_burnin=num_burnin)['p']

    return sghmc_samples, hmc_samples, posterior


def gamma_poisson_sghmc(rate, dataset_size, step_size=0.1, num_steps=4, friction=0.1, batch_size=256, num_samples=1000,
                         num_burnin=200, resample_r_freq=1):
    # posterior is over rate
    dataset, model, posterior = gamma_poisson(rate, dataset_size)
    sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size, num_steps=num_steps, friction=friction,
                                    batch_size=batch_size, num_samples=num_samples, num_burnin=num_burnin,
                                    resample_r_freq=resample_r_freq)['rate']
    hmc_samples = calculate_hmc(dataset, model, step_size=step_size, num_steps=num_steps, num_samples=num_samples,
                                num_burnin=num_burnin)['rate']

    return sghmc_samples, hmc_samples, posterior


def _2d_scatter_plot(sghmc_samples, hmc_samples, analytical_posterior):
    sghmc_plot = go.Scatter(
        x=sghmc_samples[:, 0].tolist()[0::2],
        y=sghmc_samples[:, 1].tolist()[0::2],
        mode='markers',
        marker=dict(color="darkturquoise"),
        name="SGHMC Samples"
    )

    hmc_plot = go.Scatter(
        x=hmc_samples[:, 0].tolist()[0::2],
        y=hmc_samples[:, 1].tolist()[0::2],
        mode='markers',
        marker=dict(color="hotpink"),
        name="HMC samples"
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
            start=0.05*max_analytical,
            end=max_analytical,
            size=0.1*max_analytical,
        ),
        name="Analytical posterior",
        showscale=False
    )

    fig = go.Figure(data=(analytic_plot, sghmc_plot, hmc_plot))
    fig.update_traces(showlegend=True)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))
    fig.update_yaxes(range=[y_mean - 0.3, y_mean + 0.3])
    fig.update_xaxes(range=[x_mean - 0.3, x_mean + 0.3])
    fig.show()
    fig.write_image("./figures/posterior_experiments/scatter.pdf", scale=2)


def _2d_trajectory_plot(sghmc_samples, hmc_samples, analytical_posterior):
    sghmc_plot = go.Scatter(
        x=sghmc_samples[:, 0].tolist()[-100::],
        y=sghmc_samples[:, 1].tolist()[-100::],
        mode='lines+markers',
        marker=dict(color="darkturquoise"),
        line=dict(width=1, color="darkturquoise"),
        name="SGHMC trajectory"
    )

    hmc_plot = go.Scatter(
        x=hmc_samples[:, 0].tolist()[-100::],
        y=hmc_samples[:, 1].tolist()[-100::],
        mode='lines+markers',
        marker=dict(color="hotpink"),
        line=dict(width=1, color="hotpink"),
        name="HMC trajectory"
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
            start=0.1 * max_analytical,
            end=max_analytical,
            size=0.1 * max_analytical,
        ),
        name="Analytical posterior",
        showscale=False,
    )

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=True)
    fig.add_trace(analytic_plot, row=1, col=1)
    fig.add_trace(analytic_plot, row=1, col=2)
    fig.add_trace(sghmc_plot, row=1, col=1)
    fig.add_trace(hmc_plot, row=1, col=2)
    fig.update_traces(showlegend=True)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))
    fig.update_yaxes(range=[y_mean - 0.25, y_mean + 0.25])
    fig.update_xaxes(range=[x_mean - 0.25, x_mean + 0.25])
    fig.show()
    fig.write_image("./figures/posterior_experiments/trajectories.pdf", scale=2)


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
        name="SGHMC approximation"
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
        name='HMC approximation'
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
        name='Analytical posterior'
    )

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, shared_xaxes=True)
    fig.add_trace(analytic_plot, row=1, col=1)
    fig.add_trace(sghmc_plot, row=1, col=2)
    fig.add_trace(hmc_plot, row=1, col=3)
    fig.update_traces(showlegend=True)
    fig.update_traces(showscale=False)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))
    fig.update_yaxes(range=[y_mean - 0.25, y_mean + 0.25])
    fig.update_xaxes(range=[x_mean - 0.25, x_mean + 0.25])
    fig.show()
    fig.write_image("./figures/posterior_experiments/contours.pdf", scale=2)


def _1d_plot(sghmc_samples, hmc_samples, analytical_posterior, type):
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
    plt.xlabel(type)
    plt.legend([l0, l1, l2], ["PDF", "SGHMC", "HMC"])
    plt.savefig("./figures/posterior_experiments/{}.png".format(type))
    plt.show()


def main():
    # gauss_sghmc, gauss_hmc, analytic_posterior = \
    #     _2d_gauss_sghmc(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 2000, batch_size=512,
    #                     step_size=0.02, num_steps=4, num_samples=1000, num_burnin=200, resample_r_freq=25, friction=10)
    #
    # _2d_scatter_plot(gauss_sghmc, gauss_hmc, analytic_posterior)
    # _2d_trajectory_plot(gauss_sghmc, gauss_hmc, analytic_posterior)
    # _2d_contour_plot(gauss_sghmc, gauss_hmc, analytic_posterior)

    _1d_plot(*(beta_bernoulli_sghmc(0.7, 1000, batch_size=256, step_size=0.02, num_steps=4, num_samples=1000,
                                    num_burnin=200, resample_r_freq=25, friction=10)), type="p")

    _1d_plot(*(gamma_poisson_sghmc(10, 1000, batch_size=256, step_size=0.005, num_steps=4, num_samples=1000,
                                   num_burnin=200, resample_r_freq=25, friction=10)), type="rate")


if __name__ == "__main__":
    main()