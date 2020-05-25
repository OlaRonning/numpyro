# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import logging
import os

from numpy.testing import assert_allclose
import pytest


import jax.numpy as np
from jax import jacfwd

from numpyro.infer.nmc_util import ( LimitedMemoryBFGS, ol_bfgs)

logger = logging.getLogger(__name__)


def rosenbrock_function(a, b):
    """ Bivariate function with plato at minimum (a,a**2) """

    def f(x):
        return (a - x[0]) ** 2 + b * (x[1] - (x[0] ** 2)) ** 2
    return f


def test_optimizer_banana_curve():
    a = 1  # use pytest for different values
    b = 100
    m = 10

    x = np.array([2., 2.])

    fn = rosenbrock_function(a, b)

    optimum = np.array([a, a**2])
    grad_fn = jacfwd(fn)

    lbfgs = LimitedMemoryBFGS(x, grad_fn(x), m)

    converged = False

    while not converged:
        new_x = x - lbfgs.step(x, grad_fn(x))
        assert not np.any(np.isnan(new_x)), new_x
        diff = np.abs(fn(new_x) - fn(x))
        if diff < 1e-6:
            converged = True
        x = new_x

    assert_allclose(x, optimum, atol=5e-2)

