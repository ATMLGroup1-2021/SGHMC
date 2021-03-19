import math

import torch

from experiments.analytic_posteriors import _2d_gaussian
from experiments.sghmc_experiments import _2d_gauss_sghmc, calculate_sghmc

import matplotlib.pyplot as plt
import numpy as np
import os

folder = "figures/analysis_experiments"
save = False
if save:
    os.makedirs(folder, exist_ok=True)

def save_plot(name):
    if save:
        path = os.path.join(folder, name)
        plt.savefig(f"{path}.png")
    else:
        plt.show()


def get_dist_plot(x, y, dist):
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack((xx, yy), dim=-1)
    return dist.log_prob(xy.reshape(-1, 2)).exp().reshape(len(x), len(y))


def get_heatmap(x, y, samples, bins=20):
    x = [x[0], x[-1]]
    y = [y[0], y[-1]]
    return np.histogramdd(samples, bins=bins, range=(x, y))


def step_size_experiment():
    dataset, model, posterior = _2d_gaussian(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 1000)

    x_min, x_max = 4, 6
    y_min, y_max = -9, -7

    x = torch.linspace(x_min, x_max, 256)
    y = torch.linspace(y_min, y_max, 256)
    dist_plot = get_dist_plot(x, y, posterior)

    step_sizes = [0.0025, 0.05, 0.1, 0.2]
    bins=40

    fig, axs = plt.subplots(nrows=1, ncols=len(step_sizes) + 1, figsize=(10, 2.5))
    # qcs = plt.contour(x, y, dist_plot)
    axs[0].title.set_text('Analytical')
    axs[0].imshow(dist_plot, extent=[x_min, x_max, y_min, y_max])
    axs[0].set_aspect(1.0)


    for i, step_size in enumerate(step_sizes):
        sghmc_samples = calculate_sghmc(dataset, model, step_size=step_size, num_samples=10000, resample_r_freq=25, friction=10)['mean']
        sghmc_heat = get_heatmap(x, y, sghmc_samples, bins)[0]
        axs[i+1].title.set_text(f'{step_size}')
        axs[i+1].imshow(sghmc_heat, extent=[x_min, x_max, y_min, y_max])
    fig.tight_layout()
    fig.suptitle("Effect of different step sizes")
    save_plot("step_sizes")


def num_steps_fixed_trajectorylength_experiment():
    dataset, model, posterior = _2d_gaussian(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 5000)

    x_min, x_max = 4, 6
    y_min, y_max = -9, -7

    x = torch.linspace(x_min, x_max, 256)
    y = torch.linspace(y_min, y_max, 256)
    dist_plot = get_dist_plot(x, y, posterior)

    trajectory_length = 0.2
    num_steps = [2, 4, 8, 16]
    bins=40

    fig, axs = plt.subplots(nrows=1, ncols=len(num_steps) + 1, figsize=(10, 2.5))

    axs[0].title.set_text('Analytical')
    qcs = axs[0].contour(x, y, dist_plot)
    # axs[0].imshow(dist_plot, extent=[x_min, x_max, y_min, y_max])
    axs[0].set_aspect(1)

    for i, ns in enumerate(num_steps):
        sghmc_samples = calculate_sghmc(dataset, model, step_size=trajectory_length / ns, num_steps=ns, num_samples=50, num_burnin=0)['mean']
        axs[i+1].title.set_text(f'{ns}')
        axs[i+1].plot(sghmc_samples[:, 0].numpy(), sghmc_samples[:, 1].numpy(), '--xb')
        axs[i+1].set_xlim(x_min, x_max)
        axs[i+1].set_ylim(y_min, y_max)
        axs[i+1].set_aspect(1)

    fig.tight_layout()
    fig.suptitle("Effect of different number of steps with fixed trajectory length")
    save_plot("num_steps_fixed_trajectorylength")


def num_steps_fixed_steplength_experiment():
    dataset, model, posterior = _2d_gaussian(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 5000)

    x_min, x_max = 4, 6
    y_min, y_max = -9, -7

    x = torch.linspace(x_min, x_max, 256)
    y = torch.linspace(y_min, y_max, 256)
    dist_plot = get_dist_plot(x, y, posterior)

    num_steps = [2, 4, 8, 16]
    bins=40

    fig, axs = plt.subplots(nrows=1, ncols=len(num_steps) + 1, figsize=(10, 2.5))

    axs[0].title.set_text('Analytical')
    qcs = axs[0].contour(x, y, dist_plot)
    # axs[0].imshow(dist_plot, extent=[x_min, x_max, y_min, y_max])
    axs[0].set_aspect(1)

    for i, ns in enumerate(num_steps):
        sghmc_samples = calculate_sghmc(dataset, model, step_size=0.05, num_steps=ns, num_samples=50, num_burnin=0)['mean']
        axs[i+1].title.set_text(f'{ns}')
        axs[i+1].plot(sghmc_samples[:, 0].numpy(), sghmc_samples[:, 1].numpy(), '--xb')
        axs[i+1].set_xlim(x_min, x_max)
        axs[i+1].set_ylim(y_min, y_max)
        axs[i+1].set_aspect(1)

    fig.tight_layout()
    fig.suptitle("Effect of different number of steps with fixed step length")
    save_plot("num_steps_fixed_steplength")


def resample_r_freq_friction_experiment():
    dataset, model, posterior = _2d_gaussian(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 5000)

    x_min, x_max = 4, 6
    y_min, y_max = -9, -7

    x = torch.linspace(x_min, x_max, 256)
    y = torch.linspace(y_min, y_max, 256)
    dist_plot = get_dist_plot(x, y, posterior)

    resample_r_freq = [1, 5, 10, 25]
    friction = [0.1, 1, 10]
    bins=40

    fig, axs = plt.subplots(nrows=len(friction), ncols=len(resample_r_freq), figsize=(len(resample_r_freq) * 2, len(friction) * 2 + 1.5))

    for i, f in enumerate(friction):
        for j, r in enumerate(resample_r_freq):
            sghmc_samples = calculate_sghmc(dataset, model, step_size=0.05, num_steps=4, num_samples=50, num_burnin=0, friction=f, resample_r_freq=r)['mean']
            axs[i][j].title.set_text(f'{f} / {r}')
            axs[i][j].plot(sghmc_samples[:, 0].numpy(), sghmc_samples[:, 1].numpy(), '--xb')
            axs[i][j].set_xlim(x_min, x_max)
            axs[i][j].set_ylim(y_min, y_max)
            axs[i][j].set_aspect(1)

    fig.tight_layout()
    fig.suptitle("Effect of friction / frequency of momentum resampling (step_size = 0.05)")
    save_plot("friction_resample_r_freq")


def batchsize_experiment():
    dataset, model, posterior = _2d_gaussian(torch.tensor([5., -8.]), torch.diag(torch.tensor([12., 6.])), 5000)

    x_min, x_max = 4, 6
    y_min, y_max = -9, -7

    x = torch.linspace(x_min, x_max, 256)
    y = torch.linspace(y_min, y_max, 256)
    dist_plot = get_dist_plot(x, y, posterior)

    batchsizes = [16, 64, 256, 1024]
    bins=40

    fig, axs = plt.subplots(nrows=1, ncols=len(batchsizes) + 1, figsize=(10, 2.5))

    axs[0].title.set_text('Analytical')
    qcs = axs[0].contour(x, y, dist_plot)
    # axs[0].imshow(dist_plot, extent=[x_min, x_max, y_min, y_max])
    axs[0].set_aspect(1)

    for i, bs in enumerate(batchsizes):
        sghmc_samples = calculate_sghmc(dataset, model, batch_size=bs, step_size=0.05, num_steps=4, num_samples=50, num_burnin=0, resample_r_freq=25, friction=10)['mean']
        axs[i+1].title.set_text(f'{bs}')
        axs[i+1].plot(sghmc_samples[:, 0].numpy(), sghmc_samples[:, 1].numpy(), '--xb')
        axs[i+1].set_xlim(x_min, x_max)
        axs[i+1].set_ylim(y_min, y_max)
        axs[i+1].set_aspect(1)

    fig.tight_layout()
    fig.suptitle("Effect of different batch sizes")
    save_plot("batch_sizes")


if __name__ == "__main__":
    # step_size_experiment()
    step_size_experiment()