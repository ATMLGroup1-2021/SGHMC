# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad
import itertools


def sghmc_proposal(z, r, potential_fn, step_size, friction=0.1, num_steps=1):
    r"""
    SGHMC dynamics simulation to generate new proposal state
    """
    z_next = z.copy()
    r_next = r.copy()

    for _ in range(num_steps):
        z_next, r_next = _single_step_sghmc(z_next, r_next, potential_fn, step_size, friction)

    return z_next, r_next


def _single_step_sghmc(z, r, potential_fn, step_size, friction):
    r"""
    Single step SGHMC update that modifies the `z`, `r` dicts in place.
    N.B. Implementation assumes M = I
    """

    for site_name in z:
        z[site_name] = z[site_name] + step_size * r[site_name]

    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in r:
        noise = pyro.sample(f"r_noise_{site_name}", dist.Normal(torch.zeros_like(r[site_name]), 2 * friction * step_size))
        r[site_name] = r[site_name] \
                       - step_size * friction * r[site_name] \
                       - step_size * z_grads[site_name] \
                       + noise

    return z, r


class SGHMC(MCMCKernel):
    r"""
    Stochastic Gradient Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.

    **References**

    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal

    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param float trajectory_length: Length of a MCMC trajectory. If not
        specified, it will be set to ``step_size x num_steps``. In case
        ``num_steps`` is not specified, it will be set to :math:`2\pi`.
    :param int num_steps: The number of discrete steps over which to simulate
        Hamiltonian dynamics. The state at the end of the trajectory is
        returned as the proposal. This value is always equal to
        ``int(trajectory_length / step_size)``.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 trajectory_length=None,
                 num_steps=None,
                 transforms=None):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self.step_size = step_size

        self.potential_fn = potential_fn
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan

        self._reset()
        super().__init__()

    def _kinetic_energy(self, r_unscaled):
        energy = 0.
        for site_names, value in r_unscaled.items():
            energy = energy + value.dot(value)
        return 0.5 * energy

    def _reset(self):
        self._t = 0
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._warmup_steps = None

    def _sample_r(self, name):
        r = {}
        for site_names, params in self.initial_params.items():
            # we want to sample from Normal distribution using `sample` method rather than
            # `rsample` method because the former is a bit faster
            r[site_names] = pyro.sample("{}_{}".format(name, site_names),
                                        NonreparameterizedNormal(torch.zeros_like(params, dtype=torch.float32),
                                                                 torch.ones_like(params, dtype=torch.float32)))
        return r

    @property
    def num_steps(self):
        return max(1, int(self.trajectory_length / self.step_size))

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            initial_params=self._initial_params,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace

    def setup(self, warmup_steps, *args, **kwargs):
        self.data_loader = itertools.cycle(iter(args[0]))
        args = (next(self.data_loader),)

        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        self._cache(self.initial_params)

    def cleanup(self):
        self._reset()

    def _cache(self, z):
        self._z_last = z

    def clear_cache(self):
        self._z_last = None

    def _fetch_from_cache(self):
        return self._z_last

    def sample(self, params):
        z = self._fetch_from_cache()

        if z is None:
            z = params
            self._cache(z)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            return params

        # draw new batch from data_loader
        batch = next(self.data_loader)
        # compute negative log likelihood trace
        self._initialize_model_properties((batch, ), {})

        r = self._sample_r(name="r_t={}".format(self._t))

        z_new, _ = sghmc_proposal(z, r, self.potential_fn, self.step_size, self.num_steps)

        self._t += 1

        return z_new.copy()

    def logging(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
        ])

    def diagnostics(self):
        return {"divergences": self._divergences,}