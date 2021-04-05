import math
import pickle
import sys
from typing import Optional, Callable

import pyro
import pyro.distributions as dist
import torch
from pyro import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from bnn_utils import *

pyro.set_rng_seed(101)


def test_nn(model, test_loader):
    correct = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            probs = model(batch[0])
            y_pred = torch.argmax(probs, dim=1)
            correct.append(y_pred == batch[1])
        correct = torch.cat(correct)
        acc = correct.to(dtype=torch.float32).mean().item()
    return acc


def train_nn(model, train_loader, val_loader, num_epochs=1, learning_rate=1e-3):
    len_train = len(train_loader)
    pbar = tqdm(range(num_epochs * len_train))
    losses = []
    accs = []
    acc = 0

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            prediction = model(batch[0])
            loss = F.cross_entropy(prediction, batch[1])
            loss.backward()
            optimizer.step()

            losses.append(loss)
            pbar.update()
            pbar.set_postfix_str(f"e={epoch+1}, b={(i+1):d}/{len_train:d}, loss={loss:.3f}, acc={acc:.3f}")
        acc = test_nn(model, val_loader)
        accs.append(acc)

    sys.stderr.flush()
    sys.stdout.flush()
    print()
    return losses, acc, accs


def tune_nn_hyperparameters():
    train_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=50000)
    val_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=10000, offset=50000)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)


    lrs = torch.tensor([5e-4, 1e-3, 5e-3, 1e-2])
    # lrs = torch.tensor([1e-2])

    accss = []
    best_final_acc = 0
    best_index = 0
    best_model = None
    best_lr = 0

    for i, lr in enumerate(lrs):
        model = NN(28*28, 100, 10)
        _, acc, accs = train_nn(model, train_loader, val_loader, num_epochs=20, learning_rate=lr)
        accss.append(accs)
        print(f"LR={lr}, acc={acc:.3f}")

        if acc >= best_final_acc:
            best_final_acc = acc
            best_model = model
            best_index = i
            best_lr = lr

    print(f"Best lr={best_lr}")

    np.savetxt("results/accs_svi.csv", np.array(accss[best_index]))

    test_acc = test_nn(model, test_loader)
    print(f"Final test acc={test_acc:.3f}")

    with open(f"results/nn.pkl", "wb") as f:
        pickle.dump(best_model, file=f)

if __name__ == "__main__":
    print("Choose an experiment from above to run")
    tune_nn_hyperparameters()
