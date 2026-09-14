# #################################################################################
# idaes-sdoe is part of the IDAES integrated software platform, specifically
# the Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
# #################################################################################
"""Quality-control regression tests for the native MARS response surface.

The reference predictions below were produced once, offline, by the PSUADE MARS
engine on the same seeded datasets regenerated here. They are frozen as literals
so these tests guard against drift in the native implementation without any
runtime dependency on PSUADE (or R). MARS is a greedy, discrete model-selection
method, so exact agreement is neither expected nor required; the tolerances
below reflect the level of agreement observed against the reference and are
comfortably tighter than an actual algorithmic regression would produce.
"""
from __future__ import annotations

import numpy as np

from idaes_sdoe.extras.imputation import _MarsModel


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# Reference PSUADE MARS predictions, y = (x - 0.5)**2, 40 training points.
_PSUADE_1D = [
    0.2050412, 0.1567622, 0.1084832, 0.06020412, 0.03300736, 0.02100856,
    0.00900973, -0.00298906, 0.00287804, 0.02038044, 0.03788283, 0.06457482,
    0.1034457, 0.1493524, 0.2042001,
]

# Reference PSUADE MARS predictions, 2-D interaction target + noise, 80 points.
_PSUADE_2D = [
    0.4833769, 0.3625464, -0.06327724, 0.00425306, 0.757044, 0.2295492,
    0.6960598, 0.4799267, 0.1453518, 0.6122246, 0.6611004, 0.6251588,
    0.9784603, 0.9153597, 0.06512154, 0.08475116, -0.01286852, 0.4758354,
    0.2154832, 0.1778258,
]


def test_mars_matches_psuade_reference_1d():
    rng = np.random.default_rng(0)
    n = 40
    x = np.sort(rng.random(n))
    y = (x - 0.5) ** 2
    x_test = np.linspace(0.05, 0.95, 15)

    predicted = _MarsModel().fit(x.reshape(-1, 1), y).predict(x_test.reshape(-1, 1))
    reference = np.array(_PSUADE_1D)

    assert _rmse(predicted, reference) < 0.02
    assert float(np.max(np.abs(predicted - reference))) < 0.03


def test_mars_matches_psuade_reference_2d():
    rng = np.random.default_rng(0)
    n = 80
    x_train = rng.random((n, 2))
    y = np.sin(3.0 * x_train[:, 0]) * np.cos(2.0 * x_train[:, 1]) + 0.3 * x_train[:, 0] * x_train[:, 1]
    y = y + 0.02 * rng.standard_normal(n)
    x_test = rng.random((20, 2))

    predicted = _MarsModel().fit(x_train, y).predict(x_test)
    reference = np.array(_PSUADE_2D)

    assert _rmse(predicted, reference) < 0.06
    assert float(np.max(np.abs(predicted - reference))) < 0.12
    assert float(np.corrcoef(predicted, reference)[0, 1]) > 0.99
