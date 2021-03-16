# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad
from pyro.util import optional


def sghmc_stepper(z, r, potential_fn, kinetic_grad, step_size, friction=0.1, num_steps=1, z_grads=None):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm.
    :param dict z: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param callable kinetic_grad: a function calculating gradient of kinetic energy
        w.r.t. momentum variable.
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    z_next = z.copy()
    r_next = r.copy()
    for _ in range(num_steps):
        z_next, r_next, z_grads, potential_energy = _single_sghmc_step(z_next, r_next, potential_fn, kinetic_grad, step_size, friction, z_grads)
    return z_next, r_next, z_grads, potential_energy


def _single_sghmc_step(z, r, potential_fn, kinetic_grad, step_size, friction, z_grads=None):
    r"""
    Single step velocity verlet that modifies the `z`, `r` dicts in place.
    """

    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]

    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in r:
        noise = pyro.sample(f"r_noise_{site_name}", dist.Normal(torch.zeros_like(r[site_name]), 2 * friction * step_size))
        r[site_name] = r[site_name] \
                       - step_size * friction * r_grads[site_name] \
                       - step_size * z_grads[site_name] \
                       + noise

    return z, r, z_grads, potential_energy


def kinetic_grad(r):
    return r.copy()


class SGHMC(MCMCKernel):
    r"""
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
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
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool full_mass: A flag to decide if mass matrix is dense or diagonal.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param float target_accept_prob: Increasing this value will lead to a smaller
        step size, hence the sampling will be slower and more robust. Default to 0.8.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.

    .. note:: Internally, the mass matrix will be ordered according to the order
        of the names of latent variables, not the order of their appearance in
        the model.

    Example:

        >>> true_coefs = torch.tensor([1., 2., 3.])
        >>> data = torch.randn(2000, 3)
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
        >>>
        >>> def model(data):
        ...     coefs_mean = torch.zeros(dim)
        ...     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        ...     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        ...     return y
        >>>
        >>> hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
        >>> mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
        >>> mcmc.run(data)
        >>> mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 trajectory_length=None,
                 num_steps=None,
                 transforms=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 init_strategy=init_to_uniform):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self.step_size = step_size
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._init_strategy = init_strategy

        self.potential_fn = potential_fn
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        # The following parameter is used in find_reasonable_step_size method.
        # In NUTS paper, this threshold is set to a fixed log(0.5).
        # After https://github.com/stan-dev/stan/pull/356, it is set to a fixed log(0.8).
        self._direction_threshold = math.log(0.8)  # from Stan
        self._max_sliced_energy = 1000
        self._reset()
        super().__init__()

    def _kinetic_energy(self, r_unscaled):
        energy = 0.
        for site_names, value in r_unscaled.items():
            energy = energy + value.dot(value)
        return 0.5 * energy

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None
        self._warmup_steps = None

    def _sample_r(self, name):
        r = {}
        options = {"dtype": self._potential_energy_last.dtype,
                   "device": self._potential_energy_last.device}
        for site_names, params in self.initial_params.items():
            # we want to sample from Normal distribution using `sample` method rather than
            # `rsample` method because the former is a bit faster
            r[site_names] = pyro.sample("{}_{}".format(name, site_names), NonreparameterizedNormal(torch.zeros(params.shape, **options), torch.ones(params.shape, **options)))
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
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=self._init_strategy,
            initial_params=self._initial_params,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy, z_grads)

    def cleanup(self):
        self._reset()

    def _cache(self, z, potential_energy, z_grads=None):
        self._z_last = z
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last, self._z_grads_last

    def sample(self, params):
        z, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, potential_energy, z_grads)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return params
        r = self._sample_r(name="r_t={}".format(self._t))

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = sghmc_stepper(z, r, self.potential_fn, kinetic_grad, self.step_size, self.num_steps, z_grads=z_grads)

        return z_new.copy()

    def logging(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
        ])

    def diagnostics(self):
        return {"divergences": self._divergences,}