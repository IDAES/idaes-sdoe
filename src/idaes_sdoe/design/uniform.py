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

import time
from operator import gt, lt
from typing import Iterable

import numpy as np
import pandas as pd

from ..distance import compute_distance_matrix
from ..exceptions import ConfigurationError
from ..models import DesignSetup, UniformDesignResult
from ..scaling import scale_factors_for


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize supported random-state inputs into a NumPy generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _indexed_frames(setup: DesignSetup) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return candidate and previous tables keyed by the configured index."""
    roles = setup.roles
    candidate = setup.candidate.set_index(roles.index) if roles.index else setup.candidate.copy()
    previous = (
        setup.previous.set_index(roles.index)
        if setup.previous is not None and roles.index
        else setup.previous.copy()
        if setup.previous is not None
        else None
    )
    return candidate, previous


def design_uniform(
    *,
    setup: DesignSetup,
    design_size: int,
    num_restarts: int,
    mode: str = "maximin",
    random_state: int | np.random.Generator | None = None,
) -> UniformDesignResult:
    """Construct a uniform space-filling design from a candidate set.

    Args:
        setup: Validated design setup.
        design_size: Number of rows to select.
        num_restarts: Number of random subsets to evaluate.
        mode: Uniform-space-filling criterion, either ``"maximin"`` or
            ``"minimax"``.
        random_state: Optional seed or generator for reproducible sampling.

    Returns:
        Design result containing the selected rows and criterion summary.
    """
    candidate, previous = _indexed_frames(setup)
    roles = setup.roles
    if design_size > len(candidate):
        raise ConfigurationError("Design size exceeds the number of available candidates.")

    mode_name = mode.lower()
    if mode_name not in {"maximin", "minimax"}:
        raise ConfigurationError(f"Unknown mode '{mode}'.")
    if mode_name == "maximin":
        best_value = -1.0
        reducer = np.mean
        comparator = gt
    else:
        best_value = float("inf")
        reducer = np.max
        comparator = lt

    scale = scale_factors_for(roles.inputs, setup.bounds)[roles.inputs].to_numpy()
    history_values = None if previous is None else previous[roles.inputs].to_numpy()
    rng = _generator(random_state)

    best_positions: np.ndarray | None = None
    best_matrix: np.ndarray | None = None

    start = time.time()
    indices = candidate.index.to_numpy() if roles.index else np.arange(len(candidate))
    for _ in range(num_restarts):
        selected = rng.choice(indices, size=design_size, replace=False)
        design = candidate.loc[selected]
        distance_matrix = compute_distance_matrix(
            design[roles.inputs].to_numpy(),
            scale_factors=scale,
            history_values=history_values,
        )
        minimum_distances = np.min(distance_matrix, axis=0)
        score = float(reducer(minimum_distances))
        if comparator(score, best_value):
            best_value = score
            best_positions = np.array(selected, copy=True)
            best_matrix = distance_matrix

    elapsed = time.time() - start
    if best_positions is None or best_matrix is None:
        raise RuntimeError("Uniform design search did not produce a result.")

    best_design = candidate.loc[best_positions]
    output = best_design.reset_index() if roles.index else best_design.reset_index(drop=True)
    return UniformDesignResult(
        design=output,
        selected_indices=list(best_positions.tolist()),
        criterion_value=best_value,
        distance_matrix=best_matrix,
        mode=mode_name,
        design_size=design_size,
        num_restarts=num_restarts,
        elapsed_time=elapsed,
    )


def design_uniform_batch(
    *,
    setup: DesignSetup,
    design_sizes: Iterable[int],
    num_restarts: int,
    mode: str = "maximin",
    random_state: int | np.random.Generator | None = None,
) -> list[UniformDesignResult]:
    """Run the uniform design search for several design sizes.

    Args:
        setup: Validated design setup.
        design_sizes: Iterable of design sizes to evaluate.
        num_restarts: Number of restarts to use for each size.
        mode: Uniform-space-filling criterion.
        random_state: Optional seed or generator used to derive child seeds.

    Returns:
        List of uniform-design results in the same order as ``design_sizes``.
    """
    rng = _generator(random_state)
    results: list[UniformDesignResult] = []
    for size in design_sizes:
        child_seed = int(rng.integers(0, 2**32 - 1))
        results.append(
            design_uniform(
                setup=setup,
                design_size=int(size),
                num_restarts=num_restarts,
                mode=mode,
                random_state=child_seed,
            )
        )
    return results


def estimate_uniform_runtime(
    *,
    setup: DesignSetup,
    design_sizes: Iterable[int],
    target_restarts: int,
    mode: str = "maximin",
    calibration_restarts: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, object]:
    """Estimate runtime for a uniform-design batch search.

    Args:
        setup: Validated design setup.
        design_sizes: Iterable of design sizes to evaluate.
        target_restarts: Restart count to estimate.
        mode: Uniform-space-filling criterion.
        calibration_restarts: Smaller restart count used to calibrate the
            estimate.
        random_state: Optional seed or generator.

    Returns:
        Dictionary with per-design-size and total runtime estimates in seconds.
    """
    estimates: dict[int, float] = {}
    total = 0.0
    rng = _generator(random_state)
    for size in design_sizes:
        result = design_uniform(
            setup=setup,
            design_size=int(size),
            num_restarts=calibration_restarts,
            mode=mode,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        estimated = result.elapsed_time * (target_restarts / calibration_restarts)
        estimates[int(size)] = estimated
        total += estimated
    return {
        "mode": mode,
        "target_restarts": target_restarts,
        "calibration_restarts": calibration_restarts,
        "by_design_size": estimates,
        "total_seconds": total,
    }
