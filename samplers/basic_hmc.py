"""
Custom implementation of SGHMC in Pyro
References:
     1. Pyro: Deep Universal Probabilistic Programming,
        Eli Bingham, Jonathan P Chen et al.
        Journal of Machine Learning Research, 20, 28, 2019
        https://github.com/pyro-ppl/pyro

     2. Stochastic Gradient Hamiltonian Monte Carlo,
        Tianqi Chen, Emily B. Fox et al.
        31st International Conference on Machine Learning, ICML 2014, 5, 2 2014
        https://arxiv.org/abs/1402.4102
"""

import math
from collections import OrderedDict

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan


def kinetic_grad(r):
    return r.copy()


class BasicHMC(MCMCKernel):
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
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 init_strategy=init_to_uniform):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
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
        self.step_size = step_size

        self._max_sliced_energy = 1000
        self._reset()

        super().__init__()

    def _kinetic_energy(self, r_unscaled):
        energy = 0.
        for site_names, value in r_unscaled.items():
            if len(value.shape) >= 1:
                energy += value.dot(value)
            else:
                energy += value ** 2
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
        for site_names, params in self.initial_params.items():
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
            max_plate_nesting=self._max_plate_nesting,
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
            self._mean_accept_prob = 1.
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return params
        r = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r) + potential_energy

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
                z, r, self.potential_fn, kinetic_grad, self.step_size, self.num_steps, z_grads=z_grads)
            # apply Metropolis correction.
            energy_proposal = self._kinetic_energy(r_new) + potential_energy_new
        delta_energy = energy_proposal - energy_current

        # handle the NaN case which may be the case for a diverging trajectory
        # when using a large step size.

        delta_energy = scalar_like(delta_energy, float("inf")) if torch_isnan(delta_energy) else delta_energy
        if delta_energy > self._max_sliced_energy and self._t >= self._warmup_steps:
            self._divergences.append(self._t - self._warmup_steps)

        accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(scalar_like(accept_prob, 0.),
                                                                     scalar_like(accept_prob, 1.)))
        accepted = False
        if rand < accept_prob:
            accepted = True
            z = z_new
            z_grads = z_grads_new
            self._cache(z, potential_energy_new, z_grads)

        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t

        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n
        return z.copy()

    def logging(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
            ("acc. prob", "{:.3f}".format(self._mean_accept_prob))
        ])

    def diagnostics(self):
        return {"divergences": self._divergences,
                "acceptance rate": self._accept_cnt / (self._t - self._warmup_steps)}
