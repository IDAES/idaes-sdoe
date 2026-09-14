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
"""Tests for the MaxPro design, including multi-configuration parity checks.

The reference response-surface engine's MaxPro criterion (measured on the unit
hypercube, lower is better) has these medians over stochastic runs:

    (design_size, n_inputs): median
    (8, 2): 10.976      (10, 3): 11.43      (15, 4): 15.125

The parity tests require our design to be *at least as good* as those medians
across sizes and dimensions. The reference numbers are frozen here, so the
tests need no external dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idaes_sdoe import ColumnRoles, prepare_design_setup
from idaes_sdoe.design import design_maxpro, maxpro_criterion

_REFERENCE_MEDIAN = {(8, 2): 10.976, (10, 3): 11.43, (15, 4): 15.125}


def _unit_setup(n_inputs: int):
    columns = [f"X{i}" for i in range(n_inputs)]
    candidate = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    return prepare_design_setup(candidate=candidate, roles=ColumnRoles(inputs=columns))


@pytest.mark.parametrize("design_size, n_inputs", list(_REFERENCE_MEDIAN))
def test_maxpro_matches_reference_performance(design_size, n_inputs):
    setup = _unit_setup(n_inputs)
    result = design_maxpro(
        setup=setup, design_size=design_size, num_restarts=12, random_state=0
    )
    # Lower criterion is better; require at least the reference median quality.
    assert result.criterion_value <= _REFERENCE_MEDIAN[(design_size, n_inputs)]
    assert result.mode == "maxpro"
    assert result.design.shape == (design_size, n_inputs)


def test_maxpro_criterion_helper_matches_result():
    setup = _unit_setup(3)
    result = design_maxpro(setup=setup, design_size=10, num_restarts=8, random_state=0)
    # criterion_value is measured on the unit hypercube; inputs span [0, 1] here.
    assert np.isclose(result.criterion_value, maxpro_criterion(result.design.to_numpy()))


def test_maxpro_generation_is_deterministic():
    setup = _unit_setup(3)
    first = design_maxpro(setup=setup, design_size=8, num_restarts=3, random_state=1)
    second = design_maxpro(setup=setup, design_size=8, num_restarts=3, random_state=1)
    assert np.allclose(first.design.to_numpy(), second.design.to_numpy())
    assert np.isclose(first.criterion_value, second.criterion_value)


def test_maxpro_respects_input_ranges():
    candidate = pd.DataFrame({"A": [2.0, 5.0], "B": [-1.0, 1.0]})
    setup = prepare_design_setup(candidate=candidate, roles=ColumnRoles(inputs=["A", "B"]))
    result = design_maxpro(setup=setup, design_size=8, num_restarts=3, random_state=2)
    assert result.design["A"].between(2.0, 5.0).all()
    assert result.design["B"].between(-1.0, 1.0).all()


def test_maxpro_augments_from_candidates():
    rng = np.random.default_rng(0)
    candidate = pd.DataFrame(rng.random((30, 2)), columns=["A", "B"])
    previous = pd.DataFrame(rng.random((5, 2)), columns=["A", "B"])
    setup = prepare_design_setup(
        candidate=candidate, roles=ColumnRoles(inputs=["A", "B"]), previous=previous
    )
    result = design_maxpro(setup=setup, design_size=6, random_state=0)
    assert result.selected_indices is not None
    assert len(result.selected_indices) == 6
    assert np.isfinite(result.criterion_value)
