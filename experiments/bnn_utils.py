from typing import Optional, Callable

import pyro
import math
import torch
from pyro import nn
import pyro.distributions as dist
from torch.utils.data import Dataset
from torchvision import datasets

import torch.nn.functional as F

PyroLinear = pyro.nn.PyroModule[torch.nn.Linear]


class BNN(pyro.nn.PyroModule):
    def __init__(self, input_size, hidden_size, output_size, sigma=math.sqrt(1/(500 * 2e-5)), sigmoid=True): # sigma=10.0
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

        self.activation = torch.nn.Sigmoid() if sigmoid else torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None
        x = x.view(-1, 28 * 28)
        x = self.activation(self.fc1(x))
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


class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)# output (log) softmax probabilities of each class
        return x


class MNIST_50(Dataset):

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, length=60000, offset=0) -> None:
        super().__init__()
        self.mnist = datasets.MNIST(root, train, transform, target_transform, download)
        self.length = length
        self.offset = offset

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):
        return self.mnist.__getitem__(item + self.offset)


def mode_prediction(x, predictive):
    preds = predictive(x)
    obs = preds["obs"]
    preds = torch.mode(obs, dim=0)[0]
    return preds


def mean_ll_prediction(x, predictive):
    preds = predictive(x)
    log_prob = preds["_RETURN"]
    log_prob_mean = torch.mean(log_prob, dim=0)
    preds = torch.argmax(log_prob_mean, dim=1)
    return preds


def test_posterior(predictive, test_loader):
    correct = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            y_pred = mean_ll_prediction(batch[0], predictive)
            correct.append(y_pred == batch[1])
        correct = torch.cat(correct)
        acc = correct.to(dtype=torch.float32).mean().item()
    return acc