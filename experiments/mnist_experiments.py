import os, sys, time
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

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC
from pyro.infer.mcmc.nuts import HMC


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


def sum_prediction(x, predictive):
    preds = predictive(x)
    log_prob = preds["_RETURN"]
    log_prob_sum = torch.sum(log_prob, dim=0)
    preds = torch.argmax(log_prob_sum, dim=1)
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
    predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=num_samples, return_sites=("_RETURN", "obs"))

    acc = test(predictive, test_loader)

    print()
    print(f"SVI Testing completed\n"
          f"Accuracy: {acc:.3f}")


def sample_sghmc(model, train_loader, num_samples, num_burnin, friction=0.1, step_size=0.1):
    pyro.clear_param_store()
    sghmc_kernel = SGHMC(model, step_size=step_size, num_steps=4, friction=friction)
    mcmc = MCMC(sghmc_kernel, num_samples=num_samples, warmup_steps=num_burnin, disable_progbar=False)
    mcmc.run(train_loader)
    posterior_samples = mcmc.get_samples()
    return posterior_samples


def test_sghmc(model, posterior_samples, test_loader):
    predictive = pyro.infer.Predictive(model=model, posterior_samples=posterior_samples, return_sites=("_RETURN", "obs"))

    acc = test(predictive, test_loader)

    print()
    print(f"SGHMC Testing completed\n"
          f"Accuracy: {acc:.3f}")



# Our experiments:

def compare_sghmc_and_svi_and_standard_NN():
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    bnn = BNN(28 * 28, 200, 10)

    start = time.time()
    guide, _ = train_svi(bnn, train_loader, num_epochs=1)
    test_svi(bnn, guide, test_loader, num_samples=10)
    end = time.time()
    print("Runtime:", end-start)
    print()
    print()

    start = time.time()
    posterior_samples = sample_sghmc(bnn, train_loader, num_samples=800, num_burnin=50)
    test_sghmc(bnn, posterior_samples, test_loader)
    end = time.time()
    print("Runtime:", end-start)
    print()
    print()

    class NN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, output_size)
            self.relu = torch.nn.ReLU()
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.log_softmax(x)# output (log) softmax probabilities of each class
            return x

    nn = NN(28*28, 200, 10)
    loss_function = torch.nn.NLLLoss()# negative log likelihood loss
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    start = time.time()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = nn(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    end = time.time()
    accuracy = []
    for inputs, labels in test_loader:
        outputs = torch.argmax(nn(inputs), dim=1)
        accuracy.append(sum(outputs==labels).item()/len(labels))
    print()
    print(f"Standard NN Testing completed\n"
          f"Accuracy: {sum(accuracy)/len(accuracy):.3f}")
    print("Runtime:", end-start)
    print()
    print()

def tune_bnn_hyperparameters():
    # try different configurations of batch_size and hidden_size for bnn
    for batch_size in [128, 256, 512]: # they used 500 in the paper
        for hidden_size in [80, 100, 150, 200, 250]: # they used 100 in the paper
            print("Batch size is", batch_size, "and hidden size is", hidden_size)
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            bnn = BNN(28 * 28, hidden_size, 10)

            start = time.time()
            posterior_samples = sample_sghmc(bnn, train_loader, num_samples=800, num_burnin=50)
            test_sghmc(bnn, posterior_samples, test_loader)
            end = time.time()
            print("Runtime is", end-start)
            print()
            print()

def tune_sghmc_hyperparameters():
    # try different configurations of friction and step size for bnn
    for step_size in [0.05, 0.1, 0.5, 1]:
        for friction in [0.05, 0.1, 0.15, 0.2, 0.5]:
            print("Step size is", step_size, "and friction is", friction)
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

            bnn = BNN(28 * 28, 200, 10)

            start = time.time()
            posterior_samples = sample_sghmc(bnn, train_loader, num_samples=800, num_burnin=50, friction=friction, step_size=step_size)
            test_sghmc(bnn, posterior_samples, test_loader)
            end = time.time()
            print("Runtime is", end-start)
            print()
            print()

def sghmc_on_adversarial_examples():

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    bnn = BNN(28 * 28, 200, 10)
    posterior_samples = sample_sghmc(bnn, train_loader, num_samples=800, num_burnin=50, friction=0.05, step_size=0.05)

    letters = ["a","b","c","d","e","f","g","h","i","j"]
    x_letters = np.array([np.array(Image.open("data/not-mnist/%s.png"%(l))) for l in letters]).astype(np.float32) # loads letters data
    x_letters /= np.max(x_letters, axis=(1,2), keepdims=True) # normalise
    x_letters = torch.from_numpy(x_letters)
    predictive = pyro.infer.Predictive(model=bnn, posterior_samples=posterior_samples)
    outputs = predictive(x_letters)['obs'].transpose(0,1).tolist()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.hist(outputs[i])
        plt.xlabel('class')
        plt.ylabel('frequency')
        plt.title(letters[i])
        plt.tight_layout()
        plt.savefig("figures/sghmc_on_adversarial_examples.jpg")

if __name__ == "__main__":
    print("Choose an experiment from above to run")
