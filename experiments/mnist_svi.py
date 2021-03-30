import math
import sys
from typing import Optional, Callable

import pyro
import pyro.distributions as dist
import torch
from pyro import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

pyro.set_rng_seed(101)


PyroLinear = pyro.nn.PyroModule[torch.nn.Linear]


sigma = math.sqrt(1/(500 * 2e-5))
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

        self.sigmoid = torch.nn.Sigmoid()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None
        x = x.view(-1, 28 * 28)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=x), obs=y)

        return x


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


def test(predictive, test_loader):
    len_test = len(test_loader)
    pbar = tqdm(range(len_test))

    correct = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            y_pred = mean_ll_prediction(batch[0], predictive)
            correct.append(y_pred == batch[1])
            pbar.update()
            pbar.set_postfix_str(f"b={(i+1):d}/{len_test:d}, acc={torch.cat(correct).to(dtype=torch.float32).mean().item():.3f}")
        correct = torch.cat(correct)
        acc = correct.to(dtype=torch.float32).mean().item()

    sys.stderr.flush()
    sys.stdout.flush()

    return acc


def test_svi(model, guide, test_loader, num_samples=10):
    predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=num_samples, return_sites=None)
    posterior_samples = predictive(next(iter(test_loader)))
    predictive = pyro.infer.Predictive(model=model, posterior_samples=posterior_samples, return_sites=("_RETURN", "obs"))

    acc = test(predictive, test_loader)
    return acc


def train_svi(model, guide, train_loader, test_loader, num_epochs=1, learning_rate=1e-3):
    pyro.clear_param_store()
    svi = SVI(model=model, guide=guide, optim=Adam({"lr": learning_rate}), loss=Trace_ELBO(num_particles=1))
    len_train = len(train_loader)
    pbar = tqdm(range(num_epochs * len_train))
    losses = []
    accs = []

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            loss = svi.step(batch)
            losses.append(loss)
            pbar.update()
            pbar.set_postfix_str(f"e={epoch+1}, b={(i+1):d}/{len_train:d}, loss={loss:.3f}")

        acc = test_svi(model, guide, test_loader, num_samples=800)
        print("Test Accuracy", acc)
        accs.append(acc)
    sys.stderr.flush()
    sys.stdout.flush()
    print()
    return guide, losses, accs


def tune_svi_hyperparameters(model, guide):
    train_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=50000)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # lrs = torch.tensor([5e-4, 1e-3, 5e-3, 1e-2])
    lrs = torch.tensor([5e-3])

    accss = []

    for i, lr in enumerate(lrs):
        print(f"LR={lr}")
        guide, _, accs = train_svi(model, guide, train_loader, test_loader, num_epochs=50, learning_rate=lr)
        accss.append(accs)

        best_plot = np.array(accss[torch.argmax(torch.tensor([torch.max(torch.tensor(accs)).item() for accs in accss]))])
        np.savetxt("results/accs_svi.csv", best_plot)

    sys.stderr.flush()
    sys.stdout.flush()
    print("Results")
    print(list(zip(lrs, accss)))

    best_accs = [torch.max(torch.tensor(accs)).item() for accs in accss]
    print("Overall bests")
    print(list(zip(lrs, best_accs)))

    best_plot = np.array(accss[torch.argmax(torch.tensor(best_accs))])

    np.savetxt("results/accs_svi.csv", best_plot)

    print(accss[torch.argmax(torch.tensor(best_accs))])


if __name__ == "__main__":
    print("Choose an experiment from above to run")
    bnn = BNN(28 * 28, 100, 10)
    tune_svi_hyperparameters(bnn, AutoDiagonalNormal(bnn))
    # tune_svi_hyperparameters(bnn, AutoLowRankMultivariateNormal(bnn))
