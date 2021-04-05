import pickle

from pyro.infer import MCMC, Predictive
from scipy.constants import value
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os, sys

from bnn_utils import *
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC


def test_sghmc(model, posterior_samples, test_loader):
    predictive = pyro.infer.Predictive(model=model, posterior_samples=posterior_samples, return_sites=("_RETURN", "obs"))

    acc = test_posterior(predictive, test_loader)

    print(f"Test Accuracy: {acc:.3f}")
    return acc


def sample_sghmc(model, train_loader, num_samples, num_burnin, step_size=0.1, num_steps=4, friction=0.1, resample_r_freq=1):
    pyro.clear_param_store()
    sghmc_kernel = SGHMC(model, step_size=step_size, num_steps=num_steps, friction=friction, resample_r_freq=resample_r_freq)
    mcmc = MCMC(sghmc_kernel, num_samples=num_samples, warmup_steps=num_burnin, disable_progbar=False)
    mcmc.run(train_loader)
    posterior_samples = mcmc.get_samples()
    return posterior_samples


def tune_sghmc_hyperparameters(model, num_burnin=50, num_samples=800):
    train_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=50000)
    val_dataset = MNIST_50('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]), length=10000, offset=50000)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    model = BNN(28 ** 2, 100, 10)

    print("Testing friction and resampling of r")
    print("(step_size=5e-2, num_steps=4)")

    friction = [0.1, 1, 10]
    resample_r_freq = [1, 5, 10, 25]

    best_acc = 0
    best_posterior_samples = None
    best_friction = None
    best_resample_r_freq = None

    results_friq_resample_r_freq = np.zeros((len(friction), len(resample_r_freq)))

    for i, f in enumerate(friction):
        for j, r in enumerate(resample_r_freq):
            posterior_samples = sample_sghmc(model, train_loader, step_size=0.05, num_steps=4, num_samples=num_samples, num_burnin=num_burnin, friction=f, resample_r_freq=r)
            predictive = Predictive(model, posterior_samples, return_sites=['_RETURN', 'OBS'])
            acc = test_posterior(predictive, val_loader)
            print(f"Friction={f}, Resample_r_freq={r}, acc={acc:.3f}")
            if acc >= best_acc:
                best_friction = f
                best_resample_r_freq = r
                best_acc = acc
                best_posterior_samples = posterior_samples
            results_friq_resample_r_freq[i, j] = acc

    print(f"Best Friction={best_friction}, Best Resample_r_freq={best_resample_r_freq}")

    np.savetxt("results/mnist_sghmc_friq_resample_r_freq.csv", results_friq_resample_r_freq)

    print("Testing number of steps and step sizes")

    num_steps = [2, 4, 8, 16]
    step_size = [0.0125, 0.025, 0.05, 0.1]

    best_acc = 0
    best_num_steps = None
    best_step_size = None

    results_num_steps_step_size = np.zeros((len(num_steps), len(step_size)))

    for i, num_steps_ in enumerate(num_steps):
        for j, step_size_ in enumerate(step_size):
            posterior_samples = sample_sghmc(model, train_loader, step_size=step_size_, num_steps=num_steps_, num_samples=num_samples,
                                             num_burnin=num_burnin, friction=best_friction, resample_r_freq=best_resample_r_freq)
            predictive = Predictive(model, posterior_samples, return_sites=['_RETURN', 'OBS'])
            acc = test_posterior(predictive, val_loader)
            print(f"num_steps={num_steps_}, step_size={step_size_}, accuracy={acc:.3f}")
            if acc >= best_acc:
                best_num_steps = num_steps_
                best_step_size = step_size_
                best_acc = acc
                best_posterior_samples = posterior_samples

            results_num_steps_step_size[i, j] = acc

    print(f"Best num_steps={best_num_steps}, Best step_size={best_step_size}")

    np.savetxt("results/mnist_sghmc_num_steps_step_size.csv", results_num_steps_step_size)

    predictive = Predictive(model, best_posterior_samples, return_sites=['_RETURN', 'OBS'])
    test_acc = test_posterior(predictive, test_loader)
    print(f"Final test acc={test_acc:.3f}")

    with open(f"results/posterior_samples_sghmc.pkl", "wb") as f:
        pickle.dump(best_posterior_samples, file=f)

    return best_posterior_samples, best_friction, best_resample_r_freq, best_num_steps, best_step_size


if __name__ == "__main__":
    print("Choose an experiment from above to run")
    bnn = BNN(28 * 28, 100, 10)
    posterior_samples, friction, resample_r_freq, num_steps, step_size = tune_sghmc_hyperparameters(bnn, num_samples=800)
