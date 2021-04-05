import math
import os, sys, time
import pickle

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
from pyro.infer import SVI, Trace_ELBO, MCMC, Predictive
from pyro.optim import SGD, Adam
from pyro import nn

from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC
from pyro.infer.mcmc.nuts import HMC
from bnn_utils import *


pyro.set_rng_seed(101)


def adversarial_examples():
    os.makedirs("figures/adversarial_examples", exist_ok=True)

    bnn = BNN(28 * 28, 100, 10)

    predictive_repr = Predictive(bnn, pickle.load(open("results/posterior_samples_repr.pkl", "rb")), return_sites=["_RETURN"])
    predictive_sghmc = Predictive(bnn, pickle.load(open("results/posterior_samples_sghmc.pkl", "rb")), return_sites=["_RETURN"])
    predictive_svi = Predictive(bnn, pickle.load(open("results/posterior_samples_svi.pkl", "rb")), return_sites=["_RETURN"])
    with open(f"results/nn.pkl", "rb") as f:
        nn = pickle.load(f)

    names = ["Reproduction", "SGHMC", "SVI", "NN"]
    predictors = [predictive_repr, predictive_sghmc, predictive_svi, nn]

    letters = ["a","b","c","d","e","f","g","h","i","j"]
    x_letters = np.array([np.array(Image.open("data/not-mnist/%s.png"%(l))) for l in letters]).astype(np.float32) # loads letters data
    x_letters /= np.max(x_letters, axis=(1,2), keepdims=True) # normalise
    x_letters = torch.from_numpy(x_letters)

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    test_loader_i = iter(test_loader)

    def make_prediction(predictor, x):
        if isinstance(predictor, Predictive):
            outputs = predictor(x)['_RETURN']
            # mean_nll = - outputs.mean(0)
            return outputs.exp().mean(0)
        else:
            return predictor(x)

    a_probs = {}

    for name, predictor in zip(names, predictors):
        p = make_prediction(predictor, x_letters)
        a_probs[name] = p[0]
        plt.figure()
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.bar(list(range(10)), [v.item() for v in p[i]], width=1.0)
            plt.xlabel('Class')
            if i % 5 == 0:
                plt.ylabel('Probability')
            plt.title(letters[i])
            plt.tight_layout()
        plt.savefig(f"figures/adversarial_examples/{name}_adv.jpg")

        batch = next(test_loader_i)
        p = make_prediction(predictor, batch[0])

        plt.figure()
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.bar(list(range(10)), [v.item() for v in p[i]], width=1.0)
            plt.xlabel('Class')
            if i%5 == 0:
                plt.ylabel('Probability')
            plt.title(str(batch[1][i].item()))
            plt.tight_layout()
        plt.savefig(f"figures/adversarial_examples/{name}_non_adv.jpg")

    plt.figure(figsize=(10, 2.5))
    plt.subplot(1, len(names)+1, 1)
    plt.imshow(x_letters[0])
    plt.title("Letter")
    for i, name in enumerate(names):
        plt.subplot(1, len(names)+1, i + 2)
        if i == 0:
            plt.ylabel('Probability')
        plt.bar(list(range(10)), [v.item() for v in a_probs[name]], width=1.0)
        plt.xlabel('Class')
        plt.title(name)
        plt.tight_layout()
    plt.savefig(f"figures/adversarial_examples/a_example.jpg")


if __name__ == "__main__":
    adversarial_examples()