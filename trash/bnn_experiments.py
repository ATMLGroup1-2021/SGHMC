import os, sys, torch, torchvision, pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC
from torchvision import datasets, transforms
from scipy.stats import mode

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC

# usage:
# python bnn_experiments.py hmc
# python bnn_experiments.py sghmc

# result will be saved in figures/bnn_experiments

# parameters
batch_size = 500
num_epoches = 800
burn_in = 50
input_size = 28*28
hidden_size = 100
output_size = 10

normal_mean = 1
gamma_prior_alpha = 1
gamma_prior_beta = 1

# load dataset
train_dataset, validation_dataset = torch.utils.data.random_split(datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               ])), [50000,10000])

test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                              transforms.ToTensor(),
                              ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

# define BNN
class BNN(pyro.nn.PyroModule):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.lambda_A = pyro.nn.PyroSample(dist.Gamma(gamma_prior_alpha, gamma_prior_beta))
        self.lambda_B = pyro.nn.PyroSample(dist.Gamma(gamma_prior_alpha, gamma_prior_beta))
        self.lambda_a = pyro.nn.PyroSample(dist.Gamma(gamma_prior_alpha, gamma_prior_beta))
        self.lambda_b = pyro.nn.PyroSample(dist.Gamma(gamma_prior_alpha, gamma_prior_beta))

        self.fc1 = PyroLinear(input_size, hidden_size)
        # they said in the paper the distribution is Normal and probability is proportional to exp(-lambda)
        # so I deduce that this Normal has mean=0 and variance=1/(2*lambda)
        # I think directly writing sqrt(1/(2*lambda)) will produce error, but don't know the correct way
        # TODO
        self.fc1.weight = pyro.nn.PyroSample(dist.Normal(normal_mean, sqrt(1/(2*self.lambda_A))).expand([hidden_size, input_size]))
        self.fc1.bias   = pyro.nn.PyroSample(dist.Normal(normal_mean, sqrt(1/(2*self.lambda_a))).expand([hidden_size]))

        self.fc2 = PyroLinear(hidden_size, output_size)
        self.fc2.weight = pyro.nn.PyroSample(dist.Normal(normal_mean, sqrt(1/(2*self.lambda_B))).expand([output_size, hidden_size]))
        self.fc2.bias   = pyro.nn.PyroSample(dist.Normal(normal_mean, sqrt(1/(2*self.lambda_b))).expand([output_size]))

        self.relu = torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x) # output (log) softmax probabilities of each class

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=x), obs=y)

bnn = BNN(input_size, hidden_size, output_size)

# run inference
kernel_alg = HMC if sys.argv[1] == "hmc" else SGHMC
for _ in range(num_epoches):
    # TODO
    # I am not sure whether this is correct....
    # I am fairly confindent that it is not....
    kernel = kernel_alg(bnn)
    mcmc = MCMC(kernel, num_samples = ???, warmup_steps = 100)
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()

# make prediction
accuracy = []
predictive = pyro.infer.Predictive(model=bnn, num_samples=10)
for inputs, labels in test_loader:
    prediction = torch.tensor(mode(predictive(inputs)['obs']).mode[0])
    accuracy.append(sum(prediction==labels).item()/len(labels))
print("prediction accuracies in each batch:", accuracy)
print("average prediction accuracy over all batches:", sum(accuracy)/len(accuracy))

# maybe some plots?
# TODO
