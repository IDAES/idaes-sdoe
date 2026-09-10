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
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idaes_sdoe.exceptions import ConfigurationError
from idaes_sdoe.extras.candidate_generation import generate_candidates
from idaes_sdoe.models import InputSpec


def _variable_specs(dimension: int) -> list[InputSpec]:
    """Build a list of unit-range variable input specs named X0, X1, ..."""
    return [InputSpec(name=f"X{i}", lower=-1.0, upper=1.0) for i in range(dimension)]


# orthogonal_array: valid sizes 

@pytest.mark.parametrize("num_samples, dimension", [(4, 2), (9, 3), (25, 2), (49, 4)])
def test_orthogonal_array_valid_sizes(num_samples, dimension):
    # Every size the validator accepts must also be accepted by the scipy
    # strength-2 sampler behind it.
    result = generate_candidates(
        _variable_specs(dimension),
        num_samples=num_samples,
        scheme="orthogonal_array",
        random_state=7,
    )
    assert result.samples.shape == (num_samples, dimension)
    assert result.scheme == "orthogonal_array"
    assert result.num_samples == num_samples


def test_orthogonal_array_strength_two_balance():
    # Verify the output is genuinely a strength-2 orthogonal array (index 1):
    # binning each column into p strata, every column pair sees each of the
    # p * p level combinations exactly once.
    p, dimension = 5, 3  # dimension <= p + 1
    num_samples = p * p
    result = generate_candidates(
        _variable_specs(dimension),
        num_samples=num_samples,
        scheme="orthogonal_array",
        random_state=3,
    )
    unit = (result.samples.to_numpy() + 1.0) / 2.0  # map [-1, 1] back to [0, 1)
    levels = np.clip((unit * p).astype(int), 0, p - 1)
    for i in range(dimension):
        for j in range(i + 1, dimension):
            _, counts = np.unique(levels[:, [i, j]], axis=0, return_counts=True)
            assert counts.size == p * p
            assert np.all(counts == 1)


def test_orthogonal_array_dimension_boundary():
    # p = 3 => at most p + 1 = 4 factors are allowed for num_samples = 9.
    ok = generate_candidates(
        _variable_specs(4), num_samples=9, scheme="orthogonal_array", random_state=1
    )
    assert ok.samples.shape == (9, 4)
    with pytest.raises(ConfigurationError):
        generate_candidates(
            _variable_specs(5), num_samples=9, scheme="orthogonal_array", random_state=1
        )


# orthogonal_array: invalid sizes 


@pytest.mark.parametrize(
    "num_samples, dimension, expected",
    [
        (20, 2, "9, 25"),   # ordinary bracketing
        (16, 2, "9, 25"),   # perfect square but not of a prime
        (50, 2, "49, 121"),
        (25, 7, "49"),      # dimension too large: nothing valid below
        (2, 2, "4"),        # too small: nothing valid below
    ],
)
def test_orthogonal_array_invalid_sizes(num_samples, dimension, expected):
    with pytest.raises(ConfigurationError, match=f"nearest valid sizes: {expected}"):
        generate_candidates(
            _variable_specs(dimension),
            num_samples=num_samples,
            scheme="orthogonal_array",
            random_state=1,
        )


# integration: fixed inputs & variable-dimension counting 


def test_orthogonal_array_counts_variable_dimension_only():
    # 4 variable inputs fit p = 3 (p + 1 = 4). The fixed input must not count
    # toward the OA dimension check (a total of 5 would exceed p + 1).
    specs = _variable_specs(4)
    specs.append(InputSpec(name="C", lower=0.0, upper=1.0, default=0.5, variable=False))
    result = generate_candidates(specs, num_samples=9, scheme="orthogonal_array", random_state=1)
    assert result.samples.shape == (9, 5)
    assert list(result.samples.columns) == ["X0", "X1", "X2", "X3", "C"]
    assert (result.samples["C"] == 0.5).all()
    for col in ["X0", "X1", "X2", "X3"]:
        assert result.samples[col].between(-1.0, 1.0).all()


def test_orthogonal_array_reproducible():
    a = generate_candidates(
        _variable_specs(3), num_samples=9, scheme="orthogonal_array", random_state=123
    )
    b = generate_candidates(
        _variable_specs(3), num_samples=9, scheme="orthogonal_array", random_state=123
    )
    pd.testing.assert_frame_equal(a.samples, b.samples)


# sibling schemes

@pytest.mark.parametrize(
    "scheme", ["monte_carlo", "quasi_monte_carlo", "sobol", "latin_hypercube", "metis"]
)
def test_other_schemes_shape_and_bounds(scheme):
    result = generate_candidates(_variable_specs(3), num_samples=16, scheme=scheme, random_state=11)
    assert result.samples.shape == (16, 3)
    for col in ["X0", "X1", "X2"]:
        assert result.samples[col].between(-1.0, 1.0).all()


@pytest.mark.parametrize("scheme", ["monte_carlo", "quasi_monte_carlo", "sobol", "latin_hypercube"])
def test_seeded_schemes_reproducible(scheme):
    a = generate_candidates(_variable_specs(3), num_samples=16, scheme=scheme, random_state=11)
    b = generate_candidates(_variable_specs(3), num_samples=16, scheme=scheme, random_state=11)
    pd.testing.assert_frame_equal(a.samples, b.samples)


def test_unknown_scheme_raises():
    with pytest.raises(ConfigurationError, match="Unsupported sampling scheme"):
        generate_candidates(_variable_specs(2), num_samples=8, scheme="does_not_exist")
