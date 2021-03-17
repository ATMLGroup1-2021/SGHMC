import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC
from pyro.optim import SGD
from pyro import nn

from tqdm import tqdm

from samplers.sghmc import SGHMC

pyro.set_rng_seed(101)


PyroLinear = pyro.nn.PyroModule[torch.nn.Linear]


class BNN(pyro.nn.PyroModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = PyroLinear(input_size, hidden_size)
        self.fc1.weight = nn.PyroSample(dist.Normal(0., 1.).expand((hidden_size, input_size)))
        self.fc1.bias = nn.PyroSample(dist.Normal(0., 1.).expand((hidden_size, )))

        self.fc2 = PyroLinear(hidden_size, output_size)
        self.fc2.weight = nn.PyroSample(dist.Normal(0., 1.).expand((output_size, hidden_size)))
        self.fc2.bias = nn.PyroSample(dist.Normal(0., 1.).expand((output_size, )))

        self.relu = torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=x), obs=y)

        return x


def mode_prediction(x, predictive):
    preds = predictive(x)
    obs = preds["obs"]
    preds = torch.mode(obs, dim=0)[0]
    return preds


def test(predictive, test_loader):
    len_test = len(test_loader)
    pbar = tqdm(range(len_test))

    correct = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            y_pred = mode_prediction(batch[0], predictive)
            correct.append(y_pred == batch[1])
            pbar.update()
            pbar.set_postfix_str(f"b={(i+1):d}/{len_test:d}, acc={torch.cat(correct).to(dtype=torch.float32).mean().item():.3f}")
        correct = torch.cat(correct)
        acc = correct.to(dtype=torch.float32).mean().item()
    return acc


def train_svi(model, train_loader, num_epochs=1):
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    svi = SVI(model=model, guide=guide, optim=SGD({"lr": 1e-3}), loss=Trace_ELBO(num_particles=1))
    len_train = len(train_loader)
    pbar = tqdm(range(num_epochs * len_train))
    losses = []
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            loss = svi.step(batch)
            losses.append(loss)
            pbar.update()
            pbar.set_postfix_str(f"e={epoch+1}, b={(i+1):d}/{len_train:d}, loss={loss:.3f}")
    print()
    return guide, losses


def test_svi(model, guide, test_loader, num_samples=10):
    predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=num_samples)

    acc = test(predictive, test_loader)

    print()
    print(f"SVI Testing completed\n"
          f"Accuracy: {acc:.3f}")


def sample_sghmc(model, train_loader, num_samples, num_burnin):
    pyro.clear_param_store()
    sghmc_kernel = SGHMC(model, step_size=0.1, num_steps=4)
    mcmc = MCMC(sghmc_kernel, num_samples=num_samples, warmup_steps=num_burnin)
    mcmc.run(train_loader)
    posterior_samples = mcmc.get_samples()
    return posterior_samples


def test_sghmc(model, posterior_samples, test_loader, num_samples=10):
    predictive = pyro.infer.Predictive(model=model, posterior_samples=posterior_samples, num_samples=num_samples)

    acc = test(predictive, test_loader)

    print()
    print(f"SGHMC Testing completed\n"
          f"Accuracy: {acc:.3f}")


def main():
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    bnn = BNN(28 * 28, 100, 10)

    guide, _ = train_svi(bnn, train_loader, num_epochs=1)
    test_svi(bnn, guide, test_loader)

    posterior_samples = sample_sghmc(bnn, train_loader, 500, 250)
    test_sghmc(bnn, posterior_samples, test_loader)


if __name__ == "__main__":
    main()