import pickle
import sys

import numpy as np
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bnn_utils import *

pyro.set_rng_seed(101)


def test_svi(model, guide, test_loader):
    predictive = Predictive(model=model, guide=guide, num_samples=800, return_sites=None)
    posterior_samples = predictive(next(iter(test_loader)))
    predictive = Predictive(model=model, posterior_samples=posterior_samples, return_sites=("_RETURN", "obs"))
    return test_posterior(predictive, test_loader), posterior_samples


def train_svi(model, guide, train_loader, test_loader, num_epochs=1, learning_rate=1e-3, test_samples=800):
    pyro.clear_param_store()
    svi = SVI(model=model, guide=guide, optim=Adam({"lr": learning_rate}), loss=Trace_ELBO(num_particles=1))
    len_train = len(train_loader)
    pbar = tqdm(range(num_epochs * len_train))
    losses = []
    accs = []
    acc = 0
    posterior_samples = None

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            loss = svi.step(batch)
            losses.append(loss)
            pbar.update()
            pbar.set_postfix_str(f"e={epoch+1}, b={(i+1):d}/{len_train:d}, loss={loss:.3f}, acc={acc:.3f}")
        if epoch == num_epochs-1:
            acc, posterior_samples = test_svi(model, guide, test_loader)
        accs.append(acc)
    sys.stderr.flush()
    sys.stdout.flush()
    print()
    return posterior_samples, losses, acc, accs


def tune_svi_hyperparameters(model, guide, epochs=10, test_samples=800):
    train_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=50000)
    val_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=10000, offset=50000)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    lrs = torch.tensor([5e-4, 1e-3, 5e-3, 1e-2])
    # lrs = torch.tensor([5e-3])

    accss = []
    best_final_acc = 0
    best_index = 0
    best_posterior_samples = None
    best_lr = 0

    for i, lr in enumerate(lrs):
        print(f"LR={lr}")
        posterior_samples, _, acc, accs = train_svi(model, guide, train_loader, val_loader, num_epochs=epochs, learning_rate=lr, test_samples=test_samples)
        accss.append(accs)

        if acc >= best_final_acc:
            best_final_acc = acc
            best_posterior_samples = posterior_samples
            best_index = i
            best_lr = lr
            with open(f"results/posterior_samples_svi.pkl", "wb") as f:
                pickle.dump(best_posterior_samples, file=f)

    print(f"Best lr={best_lr}")

    np.savetxt("results/accs_svi.csv", np.array(accss[best_index]))

    predictive = Predictive(model, best_posterior_samples, return_sites=['_RETURN', 'OBS'])
    test_acc = test_posterior(predictive, test_loader)
    print(f"Final test acc={test_acc:.3f}")


if __name__ == "__main__":
    print("Choose an experiment from above to run")
    bnn = BNN(28 * 28, 100, 10)
    tune_svi_hyperparameters(bnn, AutoDiagonalNormal(bnn), epochs=20, test_samples=200)

    # Requires too much memory
    # tune_svi_hyperparameters(bnn, AutoLowRankMultivariateNormal(bnn), epochs=1, test_samples=10)