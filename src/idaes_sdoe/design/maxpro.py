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
"""Maximum-projection (MaxPro) space-filling designs.

MaxPro designs (Joseph, Gul, and Ba, 2015) spread points so that projections
onto every subset of factors are well spaced, by minimizing the average of the
reciprocal products of coordinate distances:

    psi(D) = ( 1 / C(n, 2) * sum_{i<j} 1 / prod_k (x_ik - x_jk)^2 ) ^ (1/p)

Lower ``psi`` is better. This module builds a design that optimizes that
criterion directly: from-scratch designs start from a Latin hypercube and are
refined with a bound-constrained optimizer, and augmentation greedily adds the
candidate points that most improve the criterion of the combined set.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
from scipy.stats import qmc

from ..exceptions import ConfigurationError
from ..models import DesignSetup, MaxProDesignResult

_EPS = 1e-12


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize supported random-state inputs into a NumPy generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _pairwise_log_products(design: np.ndarray, power: int = 2) -> np.ndarray:
    """Return ``sum_k power * log|x_ik - x_jk|`` over all point pairs."""
    n_points, n_dims = design.shape
    log_products = np.zeros(n_points * (n_points - 1) // 2)
    for dim in range(n_dims):
        gaps = pdist(design[:, dim].reshape(-1, 1))
        log_products = log_products + power * np.log(np.maximum(gaps, _EPS))
    return log_products


def maxpro_criterion(design: np.ndarray, power: int = 2) -> float:
    """MaxPro criterion of a design on the unit hypercube (lower is better)."""
    design = np.asarray(design, dtype=float)
    n_points = design.shape[0]
    if n_points < 2:
        return float("inf")
    log_products = _pairwise_log_products(design, power)
    value = float(logsumexp(-log_products))
    return float(np.exp((value - np.log(n_points * (n_points - 1) / 2)) / design.shape[1]))


def _objective(flat: np.ndarray, n_points: int, n_dims: int) -> float:
    """Log of the summed reciprocal distance products (the quantity minimized)."""
    design = flat.reshape(n_points, n_dims)
    return float(logsumexp(-_pairwise_log_products(design)))


def _generate_design(
    n_points: int,
    n_dims: int,
    *,
    rng: np.random.Generator,
    num_restarts: int,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    """Optimize a MaxPro design on the unit hypercube over several restarts."""
    bounds = [(0.0, 1.0)] * (n_points * n_dims)
    best_design: np.ndarray | None = None
    best_value = float("inf")
    for _ in range(num_restarts):
        seed = int(rng.integers(0, 2**32 - 1))
        # A space-filling (low-discrepancy) Latin hypercube start keeps the
        # optimizer out of the poor local minima that plain random starts fall
        # into, and lets the refined design match the reference MaxPro quality.
        start = qmc.LatinHypercube(d=n_dims, optimization="random-cd", seed=seed).random(n_points)
        result = minimize(
            _objective,
            start.ravel(),
            args=(n_points, n_dims),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-11},
        )
        design = np.clip(result.x.reshape(n_points, n_dims), 0.0, 1.0)
        value = maxpro_criterion(design)
        if value < best_value:
            best_value = value
            best_design = design
    assert best_design is not None
    return best_design, best_value


def _augment_design(
    history_unit: np.ndarray,
    candidate_unit: np.ndarray,
    n_new: int,
) -> tuple[list[int], float]:
    """Greedily add candidate rows that most improve the combined criterion."""
    if len(candidate_unit) < n_new:
        raise ConfigurationError("Design size exceeds the number of available candidates.")
    current = [row for row in history_unit]
    chosen: list[int] = []
    for _ in range(n_new):
        best_position: int | None = None
        best_value = float("inf")
        for position in range(len(candidate_unit)):
            if position in chosen:
                continue
            trial = np.vstack(current + [candidate_unit[position]])
            value = maxpro_criterion(trial)
            if value < best_value:
                best_value = value
                best_position = position
        assert best_position is not None
        chosen.append(best_position)
        current.append(candidate_unit[best_position])
    return chosen, maxpro_criterion(np.vstack(current))


def design_maxpro(
    *,
    setup: DesignSetup,
    design_size: int,
    num_restarts: int = 20,
    max_iter: int = 300,
    random_state: int | np.random.Generator | None = None,
) -> MaxProDesignResult:
    """Construct a maximum-projection (MaxPro) design.

    Without previous data a MaxPro design is generated over the input ranges.
    When ``setup.previous`` is provided, new points are selected from the
    candidate set to augment the previous data.

    Args:
        setup: Validated design setup. Input ranges come from ``setup.bounds``.
        design_size: Number of design rows (new rows when augmenting).
        num_restarts: Number of optimizer restarts for from-scratch generation.
        max_iter: Maximum optimizer iterations per restart.
        random_state: Optional seed or generator.

    Returns:
        MaxPro design result. ``criterion_value`` is measured on the unit
        hypercube so it is independent of the input ranges.
    """
    roles = setup.roles
    inputs = roles.inputs
    n_dims = len(inputs)
    lowers = np.array([setup.bounds[column][0] for column in inputs], dtype=float)
    uppers = np.array([setup.bounds[column][1] for column in inputs], dtype=float)
    spans = uppers - lowers
    rng = _generator(random_state)
    start_time = time.time()

    if setup.previous is None:
        unit_design, criterion_value = _generate_design(
            design_size, n_dims, rng=rng, num_restarts=num_restarts, max_iter=max_iter
        )
        design = pd.DataFrame(unit_design * spans + lowers, columns=inputs)
        return MaxProDesignResult(
            design=design,
            criterion_value=criterion_value,
            design_size=design_size,
            mode="maxpro",
            elapsed_time=time.time() - start_time,
            selected_indices=None,
        )

    # Augmentation: scale candidate and previous inputs jointly to the unit cube.
    candidate = setup.candidate.set_index(roles.index) if roles.index else setup.candidate
    previous = setup.previous.set_index(roles.index) if roles.index else setup.previous
    candidate_values = candidate[inputs].to_numpy(dtype=float)
    previous_values = previous[inputs].to_numpy(dtype=float)
    combined = np.concatenate((candidate_values, previous_values), axis=0)
    scale_lowers = combined.min(axis=0)
    scale_spans = combined.max(axis=0) - scale_lowers
    scale_spans[scale_spans == 0] = 1.0
    candidate_unit = (candidate_values - scale_lowers) / scale_spans
    previous_unit = (previous_values - scale_lowers) / scale_spans

    positions, criterion_value = _augment_design(previous_unit, candidate_unit, design_size)
    labels = candidate.index.to_numpy()[positions]
    selected = candidate.iloc[positions].reset_index() if roles.index else candidate.iloc[positions].reset_index(drop=True)
    return MaxProDesignResult(
        design=selected,
        criterion_value=criterion_value,
        design_size=design_size,
        mode="maxpro",
        elapsed_time=time.time() - start_time,
        selected_indices=list(labels.tolist()),
    )
