import math
import os, sys, time
import pickle
from typing import Optional, Callable

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
from pyro.optim import SGD, Adam
from pyro import nn

from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC
from pyro.infer.mcmc.nuts import HMC


pyro.set_rng_seed(101)


PyroLinear = pyro.nn.PyroModule[torch.nn.Linear]


sigma = math.sqrt(1/(500 * 2e-5)) # = 10.0
# sigma = 1


class BNN(pyro.nn.PyroModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = PyroLinear(input_size, hidden_size)
        self.fc1.weight = nn.PyroSample(dist.Normal(0., sigma).expand((hidden_size, input_size)))
        self.fc1.bias = nn.PyroSample(dist.Normal(0., sigma).expand((hidden_size, )))

        self.fc2 = PyroLinear(hidden_size, output_size)
        self.fc2.weight = nn.PyroSample(dist.Normal(0., sigma).expand((output_size, hidden_size)))
        self.fc2.bias = nn.PyroSample(dist.Normal(0., sigma).expand((output_size, )))

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


def bnn_gamma(batch):
    if len(batch) == 2:
        x, y = batch
    else:
        x = batch
        y = None

    lambda_A = pyro.sample("lambda_A", dist.Gamma(1, 1))
    lambda_a = pyro.sample("lambda_a", dist.Gamma(1, 1))
    lambda_B = pyro.sample("lambda_B", dist.Gamma(1, 1))
    lambda_b = pyro.sample("lambda_b", dist.Gamma(1, 1))

    A = pyro.sample("fc1.weight", dist.Normal(torch.zeros(784, 100), torch.sqrt(1 / lambda_A)))
    a = pyro.sample("fc1.bias", dist.Normal(torch.zeros(1, 100), torch.sqrt(1 / lambda_a)))
    B = pyro.sample("fc2.weight", dist.Normal(torch.zeros(100, 10), torch.sqrt(1 / lambda_B)))
    b = pyro.sample("fc2.bias", dist.Normal(torch.zeros(1, 10), torch.sqrt(1 / lambda_b)))

    x = x.view(-1, 28 * 28)
    x = F.relu(x @ A + a)
    x = x @ B + b
    x = F.log_softmax(x)

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=x), obs=y)

    return x


def mode_prediction(x, predictive):
    preds = predictive(x)
    obs = preds["obs"]
    preds = torch.mode(obs, dim=0)[0]
    return preds


def sum_prediction(x, predictive):
    preds = predictive(x)
    log_prob = preds["_RETURN"]
    log_prob_mean = torch.mean(log_prob, dim=0)
    preds = torch.argmax(log_prob_mean, dim=1)
    return preds


def test(predictive, test_loader):
    len_test = len(test_loader)
    pbar = tqdm(range(len_test))

    correct = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            y_pred = sum_prediction(batch[0], predictive)
            correct.append(y_pred == batch[1])
            pbar.update()
            pbar.set_postfix_str(f"b={(i+1):d}/{len_test:d}, acc={torch.cat(correct).to(dtype=torch.float32).mean().item():.3f}")
        correct = torch.cat(correct)
        acc = correct.to(dtype=torch.float32).mean().item()
    return acc


def efficient_test(model, sample, test_loader, previous_prediction_means, num_prev_predictions):
    predictive = pyro.infer.Predictive(model=model, posterior_samples=sample, return_sites=("_RETURN", "obs"))

    alpha = num_prev_predictions / (num_prev_predictions + 1)

    correct = []

    len_test = len(test_loader)

    if previous_prediction_means is None:
        previous_prediction_means = [None for _ in range(len_test)]

    pbar = tqdm(range(len_test))
    for i, batch in enumerate(test_loader):
        if previous_prediction_means[i] is None:
            previous_prediction_means[i] = predictive(batch[0])["_RETURN"][0]
        else:
            previous_prediction_means[i] = alpha * previous_prediction_means[i] + (1 - alpha) * predictive(batch[0])["_RETURN"][0]
        y_pred = torch.argmax(previous_prediction_means[i], dim=-1)
        correct.append(y_pred == batch[1])
        pbar.update()
        pbar.set_postfix_str(f"b={(i + 1):d}/{len_test:d}, acc={torch.cat(correct).to(dtype=torch.float32).mean().item():.3f}")
    correct = torch.cat(correct)

    return correct.to(dtype=torch.float32).mean(), previous_prediction_means


def initial_weights(bnn, sigma, add_lambdas=False):
    fc1_weight = torch.randn(bnn.fc1.weight.shape) * sigma
    fc1_bias = torch.zeros_like(bnn.fc1.bias)
    fc2_weight = torch.randn(bnn.fc2.weight.shape) * sigma
    fc2_bias = torch.zeros_like(bnn.fc2.bias)
    params = {
        "fc1.weight": fc1_weight,
        "fc1.bias": fc1_bias,
        "fc2.weight": fc2_weight,
        "fc2.bias": fc2_bias
    }
    if add_lambdas:
        params["fc1.weight"] = fc1_weight.T
        params["fc2.weight"] = fc2_weight.T

        params["lambda_A"] = torch.ones(1)
        params["lambda_a"] = torch.ones(1)
        params["lambda_B"] = torch.ones(1)
        params["lambda_b"] = torch.ones(1)
    return params


def manual_init_sample_sghmc(model, train_loader, z, r, num_samples, num_burnin, friction=0.1, step_size=0.1,
                             resample_r_freq=1, resample_r=False, noise_scale=0.01, num_steps=4, mult_step_size_on_r=False):
    pyro.clear_param_store()
    sghmc_kernel = SGHMC(model, step_size=step_size, num_steps=num_steps, friction=friction, resample_r_freq=resample_r_freq, resample_r=resample_r, noise_scale=noise_scale, mult_step_size_on_r=mult_step_size_on_r)
    sghmc_kernel.manual_initialization(z, r)
    mcmc = MCMC(sghmc_kernel, num_samples=num_samples, warmup_steps=num_burnin, disable_progbar=False)
    mcmc.run(train_loader)
    posterior_samples = mcmc.get_samples()
    z, r = sghmc_kernel._fetch_from_cache()
    return posterior_samples, z, r


def test_sghmc(model, posterior_samples, test_loader):
    predictive = pyro.infer.Predictive(model=model, posterior_samples=posterior_samples, return_sites=("_RETURN", "obs"))

    acc = test(predictive, test_loader)

    print()
    print(f"SGHMC Testing completed\n"
          f"Accuracy: {acc:.3f}")
    return acc


class MNIST_50(Dataset):

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, length=60000) -> None:
        super().__init__()
        self.mnist = datasets.MNIST(root, train, transform, target_transform, download)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):
        return self.mnist.__getitem__(item)


def sghmc_reproduction(batch_size=500, num_epochs=800, suffix=""):
    train_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=50000)
    # train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_samples = int(np.floor(train_dataset.__len__()/batch_size))

    bnn = BNN(28 * 28, 100, 10)

    z = initial_weights(bnn, 0.01)
    # z = None
    r = None

    posterior_samples = None
    previous_predictions = None
    num_previous_predictions = 0

    burnin = 50
    save_freq = 25

    single_accs = []
    accs = []

    start = time.time()
    for i in range(num_epochs):
        print("Epoch {}:".format(i))
        if i < 1:
            noise_scale = 0
        else:
            noise_scale = 0.1 / len(train_dataset) # Number of samples in training set
        # sample, z, r = manual_init_sample_sghmc(bnn, train_loader, z, r, num_samples=1, num_burnin=num_samples-1, num_steps=1, step_size=1e-2, resample_r_freq=10, resample_r=False, friction=10, noise_scale=noise_scale)
        sample, z, r = manual_init_sample_sghmc(bnn, train_loader, z, r, num_samples=1, num_burnin=num_samples-1, num_steps=1, step_size=2e-4, resample_r=False, friction=50, noise_scale=noise_scale, mult_step_size_on_r=False)

        print("Current sample")
        single_accs.append(test_sghmc(bnn, sample, test_loader))

        if i >= burnin:
            if posterior_samples is None:
                posterior_samples = sample
            else:
                for key in posterior_samples.keys():
                    posterior_samples[key] = torch.cat((posterior_samples[key], sample[key]), dim=0)
            print("Num Probosals", len(posterior_samples[next(iter(posterior_samples.keys()))]))
            print("All samples")
            acc, previous_predictions = efficient_test(bnn, sample, test_loader, previous_predictions, num_previous_predictions)
            accs.append(acc)
            print("Overall accuracy", acc)

            num_previous_predictions += 1

            single_accs_ = np.array(single_accs)
            accs_ = np.array(accs)
            np.savetxt(f"results/single_accs{suffix}.csv", single_accs_)
            np.savetxt(f"results/accs{suffix}.csv", accs_)

            if ((i + 1) % save_freq) == 0:
                with open(f"results/posterior_samples{suffix}.pkl", "wb") as f:
                    pickle.dump(posterior_samples, file=f)

    end = time.time()
    print("Runtime:", end - start)
    print()
    print()

    print(single_accs)
    print(accs)


if __name__ == "__main__":
    sghmc_reproduction(500, 800, "_50")
