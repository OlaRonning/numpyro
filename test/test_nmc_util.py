# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import logging
import os

from numpy.testing import assert_allclose
import pytest


import numpyro.distributions as dist
from numpyro.infer.nmc_util import ( LimitedMemoryBFGS, ol_bfgs )
from numpyro.util import control_flow_prims_disabled, fori_loop, optional

logger = logging.getLogger(__name__)

def test_lbfgs():
    pass
