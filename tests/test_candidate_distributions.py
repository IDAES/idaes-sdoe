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
"""Distribution correctness tests for candidate generation.

Each generated column is checked against the analytic distribution it is meant
to draw from (theoretical moments/quantiles built independently in the test, and
a Kolmogorov-Smirnov goodness-of-fit test against the closed-form CDF) rather
than against any reference implementation's output. Samples are drawn with the
``monte_carlo`` scheme (i.i.d.) and a fixed seed, so the checks are meaningful
and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from idaes_sdoe.exceptions import ConfigurationError
from idaes_sdoe.extras.candidate_generation import (
    generate_candidates,
    load_template_specs,
    specs_from_previous,
)
from idaes_sdoe.models import InputSpec

_N = 20000


def _samples(spec: InputSpec, n: int = _N, seed: int = 0) -> np.ndarray:
    result = generate_candidates([spec], num_samples=n, scheme="monte_carlo", random_state=seed)
    return result.samples[spec.name].to_numpy()


# --- distribution shape correctness (vs analytic ground truth) ---------------


def test_uniform_distribution():
    s = _samples(InputSpec("x", 2.0, 5.0, distribution="uniform"))
    assert s.min() >= 2.0 and s.max() <= 5.0
    assert abs(s.mean() - 3.5) < 0.05
    assert st.kstest(s, st.uniform(loc=2.0, scale=3.0).cdf).statistic < 0.02


def test_normal_distribution():
    spec = InputSpec("x", -20.0, 20.0, distribution="normal", parameters={"mean": 5.0, "sd": 2.0})
    s = _samples(spec)
    assert abs(s.mean() - 5.0) < 0.05
    assert abs(s.std() - 2.0) < 0.05
    assert st.kstest(s, st.norm(loc=5.0, scale=2.0).cdf).statistic < 0.02


def test_lognormal_distribution():
    spec = InputSpec("x", 0.0, 60.0, distribution="lognormal", parameters={"sigma": 0.5, "mean": 0.0})
    s = _samples(spec)
    logs = np.log(s)
    assert abs(logs.mean() - 0.0) < 0.05
    assert abs(logs.std() - 0.5) < 0.05
    assert st.kstest(s, st.lognorm(s=0.5, scale=1.0).cdf).statistic < 0.03


def test_triangular_distribution():
    spec = InputSpec("x", 2.0, 5.0, distribution="triangular", parameters={"mode": 3.0})
    s = _samples(spec)
    assert s.min() >= 2.0 and s.max() <= 5.0
    assert abs(s.mean() - (2.0 + 3.0 + 5.0) / 3.0) < 0.05
    c = (3.0 - 2.0) / (5.0 - 2.0)
    assert st.kstest(s, st.triang(c=c, loc=2.0, scale=3.0).cdf).statistic < 0.02


def test_beta_distribution():
    spec = InputSpec("x", 0.0, 1.0, distribution="beta", parameters={"a": 2.0, "b": 5.0})
    s = _samples(spec)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert abs(s.mean() - 2.0 / 7.0) < 0.02
    assert st.kstest(s, st.beta(2.0, 5.0).cdf).statistic < 0.02


def test_gamma_distribution():
    spec = InputSpec("x", 0.0, 1000.0, distribution="gamma", parameters={"shape": 2.0, "scale": 1.5})
    s = _samples(spec)
    assert s.min() >= 0.0
    assert abs(s.mean() - 3.0) < 0.1      # shape * scale
    assert abs(s.var() - 4.5) < 0.6       # shape * scale**2
    assert st.kstest(s, st.gamma(a=2.0, scale=1.5).cdf).statistic < 0.02


def test_exponential_distribution():
    spec = InputSpec("x", 0.0, 1000.0, distribution="exponential", parameters={"scale": 2.0})
    s = _samples(spec)
    assert s.min() >= 0.0
    assert abs(s.mean() - 2.0) < 0.1
    assert st.kstest(s, st.expon(scale=2.0).cdf).statistic < 0.02


# --- bug-fix regressions -----------------------------------------------------


def test_normal_zero_default_mean_is_respected():
    # default=0.0 with no explicit "mean" must center at 0.0, not the midpoint
    # of [lower, upper] (which is 10 here). Guards the falsy-0.0 fallback bug.
    spec = InputSpec("x", -10.0, 30.0, default=0.0, distribution="normal", parameters={"sd": 2.0})
    s = _samples(spec)
    assert abs(s.mean() - 0.0) < 0.1


def test_gamma_lower_shifts_support_not_clip():
    # lower=10 must shift the gamma to start at 10, not clip a 0-anchored gamma
    # (which would pile all mass at 10).
    spec = InputSpec("x", 10.0, 1000.0, distribution="gamma", parameters={"shape": 2.0, "scale": 1.5})
    s = _samples(spec)
    assert s.min() >= 10.0
    assert abs(s.mean() - 13.0) < 0.15         # lower + shape * scale
    assert s.std() > 1.0                         # genuinely spread, not a spike
    assert (s <= 10.0 + 1e-9).mean() < 0.001     # no pile-up at the bound


def test_exponential_lower_shifts_support_not_clip():
    spec = InputSpec("x", 5.0, 1000.0, distribution="exponential", parameters={"scale": 2.0})
    s = _samples(spec)
    assert s.min() >= 5.0
    assert abs(s.mean() - 7.0) < 0.15            # lower + scale
    assert (s <= 5.0 + 1e-9).mean() < 0.001


def test_lognormal_lower_shifts_support():
    spec = InputSpec("x", 5.0, 60.0, distribution="lognormal", parameters={"sigma": 0.5, "mean": 0.0})
    s = _samples(spec)
    assert s.min() >= 5.0
    logs = np.log(s - 5.0)
    assert abs(logs.mean() - 0.0) < 0.06
    assert abs(logs.std() - 0.5) < 0.06


# --- mechanics ---------------------------------------------------------------


def test_fixed_input_is_constant_and_order_preserved():
    specs = [
        InputSpec("v", 0.0, 1.0),
        InputSpec("c", 0.0, 10.0, default=4.0, variable=False),
    ]
    result = generate_candidates(specs, num_samples=50, scheme="monte_carlo", random_state=0)
    assert result.samples.shape == (50, 2)
    assert list(result.samples.columns) == ["v", "c"]
    assert (result.samples["c"] == 4.0).all()


def test_fixed_input_defaults_to_lower_without_default():
    specs = [InputSpec("v", 0.0, 1.0), InputSpec("c", 3.0, 10.0, variable=False)]
    result = generate_candidates(specs, num_samples=10, random_state=0)
    assert (result.samples["c"] == 3.0).all()


def test_reproducible_and_seed_sensitive():
    spec = InputSpec("x", 0.0, 1.0, distribution="normal", parameters={"mean": 0.5, "sd": 0.1})
    a = _samples(spec, n=200, seed=1)
    b = _samples(spec, n=200, seed=1)
    c = _samples(spec, n=200, seed=2)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# --- error handling ----------------------------------------------------------


def test_unsupported_distribution_raises():
    spec = InputSpec("x", 0.0, 1.0, distribution="weibull", parameters={"shape": 1.0})
    with pytest.raises(ConfigurationError):
        generate_candidates([spec], num_samples=5, random_state=0)


def test_missing_required_parameter_raises():
    spec = InputSpec("x", -5.0, 5.0, distribution="normal")  # no "sd"
    with pytest.raises((KeyError, ConfigurationError)):
        generate_candidates([spec], num_samples=5, random_state=0)


# --- spec builders -----------------------------------------------------------


def test_load_template_specs(tmp_path):
    path = tmp_path / "template.csv"
    pd.DataFrame({"A": [0.0, 10.0, 5.0], "B": [-1.0, 1.0, 0.0]}).to_csv(path, index=False)
    specs = {s.name: s for s in load_template_specs(path)}
    assert specs["A"].lower == 0.0 and specs["A"].upper == 10.0 and specs["A"].default == 5.0
    assert specs["B"].lower == -1.0 and specs["B"].upper == 1.0 and specs["B"].default == 0.0


def test_load_template_specs_requires_min_and_max(tmp_path):
    path = tmp_path / "one_row.csv"
    pd.DataFrame({"A": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ConfigurationError):
        load_template_specs(path)


def test_specs_from_previous_uses_min_max_mean_and_subset():
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [0.0, 0.0, 0.0]})
    specs = specs_from_previous(frame, columns=["A", "B"])
    assert [s.name for s in specs] == ["A", "B"]
    by_name = {s.name: s for s in specs}
    assert by_name["A"].lower == 1.0 and by_name["A"].upper == 3.0 and by_name["A"].default == 2.0
    assert by_name["B"].lower == 10.0 and by_name["B"].upper == 30.0 and by_name["B"].default == 20.0
