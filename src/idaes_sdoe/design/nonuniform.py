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

import numpy as np
import pandas as pd

from ..distance import compute_distance_matrix, minimum_distance_summary
from ..exceptions import ConfigurationError
from ..models import DesignSetup, NonUniformDesignResult
from ..scaling import scale_columns, scale_weights


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize supported random-state inputs into a NumPy generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _indexed_frames(setup: DesignSetup) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return candidate and previous tables keyed by the configured index."""
    roles = setup.roles
    if not roles.weight:
        raise ConfigurationError("A weight column is required for non-uniform design.")
    candidate = setup.candidate.set_index(roles.index) if roles.index else setup.candidate.copy()
    previous = (
        setup.previous.set_index(roles.index)
        if setup.previous is not None and roles.index
        else setup.previous.copy()
        if setup.previous is not None
        else None
    )
    return candidate, previous


def _scaled_numeric_frames(
    setup: DesignSetup,
    *,
    mwr: int,
    scale_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None]:
    """Prepare raw and scaled numeric tables for the NUSF search.

    Args:
        setup: Validated design setup.
        mwr: Maximum weight ratio.
        scale_method: Weight-scaling method.

    Returns:
        Candidate table, previous-data table, scaled candidate table, and
        scaled previous-data table.
    """
    candidate, previous = _indexed_frames(setup)
    roles = setup.roles
    used = roles.inputs + [roles.weight]
    candidate_used = candidate[used].copy()
    previous_used = previous[used].copy() if previous is not None else None

    combined = candidate_used if previous_used is None else pd.concat([candidate_used, previous_used], axis=0)
    scaled = combined.copy()
    scaled[roles.inputs] = scale_columns(scaled[roles.inputs], columns=roles.inputs, bounds=setup.bounds)
    scaled[roles.weight] = scale_weights(
        scaled[roles.weight].to_numpy(),
        method=scale_method,
        mwr=mwr,
    )
    if previous_used is None:
        return candidate, previous, scaled, None
    return candidate, previous, scaled.iloc[: len(candidate_used)].copy(), scaled.iloc[len(candidate_used) :].copy()


def _weighted_distance_matrix(
    design: np.ndarray,
    *,
    input_indices: list[int],
    weight_index: int,
    history: np.ndarray | None,
) -> np.ndarray:
    """Compute a weighted distance matrix for NUSF search updates.

    Args:
        design: Current scaled design array.
        input_indices: Positions of input columns inside ``design``.
        weight_index: Position of the scaled weight column.
        history: Optional scaled previous-data array.

    Returns:
        Weighted distance matrix including history rows when provided.
    """
    inputs = design[:, input_indices]
    weights = design[:, weight_index]
    if history is None:
        return compute_distance_matrix(inputs, weights=weights)
    matrix = compute_distance_matrix(
        inputs,
        weights=weights,
        history_values=history[:, input_indices],
        history_weights=history[:, weight_index],
    )
    history_size = len(history)
    matrix[-history_size:, -history_size:] = np.max(matrix)
    return matrix


def _update_min_distance(
    current_design: np.ndarray,
    candidates: np.ndarray,
    *,
    input_indices: list[int],
    weight_index: int,
    current_minimum: float,
    current_points: np.ndarray,
    current_ties: int,
    distance_matrix: np.ndarray,
    history: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray, int | None, int | None, bool]:
    """Attempt a single improving swap for the NUSF search state.

    Args:
        current_design: Current scaled design array.
        candidates: Full scaled candidate array.
        input_indices: Positions of input columns.
        weight_index: Position of the weight column.
        current_minimum: Current criterion value.
        current_points: Point indices involved in minimum-distance ties.
        current_ties: Number of current tied pairs.
        distance_matrix: Current weighted distance matrix.
        history: Optional scaled previous-data array.
        rng: Random generator used to break ties.

    Returns:
        Updated design state, criterion summaries, and a flag indicating whether
        an improving swap was found.
    """
    def refresh_distance(row: np.ndarray, design: np.ndarray, matrix: np.ndarray, design_position: int) -> np.ndarray:
        """Refresh one row and column of the weighted distance matrix."""
        inputs = design[:, input_indices] if history is None else np.concatenate((design[:, input_indices], history[:, input_indices]))
        weights = design[:, weight_index] if history is None else np.concatenate((design[:, weight_index], history[:, weight_index]))
        values = np.sum(np.square(row[input_indices] - inputs), axis=1)
        values = values * row[weight_index] * weights
        values[design_position] = 9999.0
        updated = np.array(matrix, copy=True)
        updated[design_position, :] = values
        updated[:, design_position] = values
        return updated

    def try_swap(
        design_min_position: int,
        candidate_position: int,
    ) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray, int, int]:
        """Evaluate the effect of swapping one design row with one candidate."""
        design = np.array(current_design, copy=True)
        candidate_row = candidates[candidate_position]
        replace_at = design_points[design_min_position]
        design[replace_at] = candidate_row
        updated_matrix = refresh_distance(candidate_row, design, distance_matrix, replace_at)
        minimum, points, ties = minimum_distance_summary(updated_matrix)
        return design, minimum, points, ties, updated_matrix, candidate_position, replace_at

    design_points = current_points[current_points < len(current_design)]
    n_points = len(design_points)
    n_candidates = len(candidates)
    improvements = np.zeros((n_points, n_candidates))
    ties = np.zeros((n_points, n_candidates))

    for design_min_position in range(n_points):
        for candidate_position in range(n_candidates):
            _, improvements[design_min_position, candidate_position], _, ties[design_min_position, candidate_position], _, _, _ = try_swap(
                design_min_position, candidate_position
            )

    best_value = np.max(improvements)
    best_pairs = np.argwhere(improvements == best_value)
    if best_value > current_minimum:
        pair = best_pairs[int(rng.integers(0, len(best_pairs)))]
        return (*try_swap(int(pair[0]), int(pair[1])), True)
    if best_value == current_minimum:
        better_ties = np.argwhere(ties[best_pairs[:, 0], best_pairs[:, 1]] < current_ties).flatten()
        if len(better_ties) > 0:
            pair = best_pairs[int(rng.choice(better_ties))]
            return (*try_swap(int(pair[0]), int(pair[1])), True)
    return current_design, current_minimum, current_points, current_ties, distance_matrix, None, None, False


def design_nonuniform(
    *,
    setup: DesignSetup,
    design_size: int,
    num_restarts: int,
    mwr: int,
    scale_method: str = "direct_mwr",
    random_state: int | np.random.Generator | None = None,
    max_iterations: int = 100,
) -> NonUniformDesignResult:
    """Construct a non-uniform space-filling design.

    Args:
        setup: Validated design setup with a weight column.
        design_size: Number of rows to select.
        num_restarts: Number of random restarts.
        mwr: Maximum weight ratio used for scaled weights.
        scale_method: Weight-scaling method.
        random_state: Optional seed or generator.
        max_iterations: Maximum local-improvement iterations per restart.

    Returns:
        Design result containing raw and scaled selected rows plus criterion
        details.
    """
    candidate, _previous, candidate_scaled, previous_scaled = _scaled_numeric_frames(
        setup,
        mwr=mwr,
        scale_method=scale_method,
    )
    if design_size > len(candidate_scaled):
        raise ConfigurationError("Design size exceeds the number of available candidates.")

    roles = setup.roles
    used_columns = roles.inputs + [roles.weight]
    candidate_array = candidate_scaled[used_columns].to_numpy()
    history_array = previous_scaled[used_columns].to_numpy() if previous_scaled is not None else None
    input_indices = list(range(len(roles.inputs)))
    weight_index = len(roles.inputs)
    rng = _generator(random_state)

    best_positions: np.ndarray | None = None
    best_scaled_design: np.ndarray | None = None
    best_value = 0.0
    best_ties = 0
    best_matrix: np.ndarray | None = None

    start = time.time()
    for _ in range(num_restarts):
        weights = candidate_array[:, weight_index]
        probabilities = weights / np.sum(weights)
        selected_positions = rng.choice(len(candidate_array), size=design_size, replace=False, p=probabilities)
        design = candidate_array[selected_positions]
        distance_matrix = _weighted_distance_matrix(
            design,
            input_indices=input_indices,
            weight_index=weight_index,
            history=history_array,
        )
        minimum, points, ties = minimum_distance_summary(distance_matrix)

        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            design, minimum, points, ties, distance_matrix, added, removed, improved = _update_min_distance(
                design,
                candidate_array,
                input_indices=input_indices,
                weight_index=weight_index,
                current_minimum=minimum,
                current_points=points,
                current_ties=ties,
                distance_matrix=distance_matrix,
                history=history_array,
                rng=rng,
            )
            if improved and added is not None and removed is not None:
                selected_positions[removed] = added
            iterations += 1

        if (minimum > best_value) or (minimum == best_value and (best_ties == 0 or ties < best_ties)):
            best_positions = np.array(selected_positions, copy=True)
            best_scaled_design = np.array(design, copy=True)
            best_value = float(minimum)
            best_ties = int(ties)
            best_matrix = distance_matrix

    elapsed = time.time() - start
    if best_positions is None or best_scaled_design is None or best_matrix is None:
        raise RuntimeError("Non-uniform design search did not produce a result.")

    selected_labels = candidate.index.to_numpy()[best_positions]
    design = candidate.loc[selected_labels].reset_index() if roles.index else candidate.iloc[best_positions].reset_index(drop=True)
    scaled_design = pd.DataFrame(best_scaled_design, columns=used_columns, index=selected_labels)
    if roles.index:
        scaled_design = scaled_design.reset_index().rename(columns={"index": roles.index})
    else:
        scaled_design = scaled_design.reset_index(drop=True)

    return NonUniformDesignResult(
        design=design,
        scaled_design=scaled_design,
        selected_indices=list(selected_labels.tolist()),
        criterion_value=best_value,
        distance_matrix=best_matrix,
        mode="maximin",
        design_size=design_size,
        num_restarts=num_restarts,
        mwr=mwr,
        elapsed_time=elapsed,
        tie_count=best_ties,
    )


def estimate_nonuniform_runtime(
    *,
    setup: DesignSetup,
    design_size: int,
    target_restarts: int,
    mwr_values: list[int],
    scale_method: str = "direct_mwr",
    calibration_restarts: int = 10,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, object]:
    """Estimate runtime for a non-uniform-design search over MWR values.

    Args:
        setup: Validated design setup.
        design_size: Number of rows to select.
        target_restarts: Restart count to estimate.
        mwr_values: MWR settings to evaluate.
        scale_method: Weight-scaling method.
        calibration_restarts: Smaller restart count used to calibrate the
            estimate.
        random_state: Optional seed or generator.

    Returns:
        Dictionary with per-MWR and total runtime estimates in seconds.
    """
    rng = _generator(random_state)
    estimates: dict[int, float] = {}
    total = 0.0
    for mwr in mwr_values:
        result = design_nonuniform(
            setup=setup,
            design_size=design_size,
            num_restarts=calibration_restarts,
            mwr=mwr,
            scale_method=scale_method,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        estimated = result.elapsed_time * (target_restarts / calibration_restarts)
        estimates[mwr] = estimated
        total += estimated
    return {
        "mode": "maximin",
        "design_size": design_size,
        "target_restarts": target_restarts,
        "scale_method": scale_method,
        "by_mwr": estimates,
        "total_seconds": total,
    }
