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

import itertools
import time

import numpy as np
import pandas as pd

from ..distance import compute_distance_matrix, minimum_distance_summary
from ..exceptions import ConfigurationError
from ..models import DesignSetup, InputResponseDesignResult
from ..scaling import inverse_scale_columns, scale_columns


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize supported random-state inputs into a NumPy generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _compute_distance(values: np.ndarray, history: np.ndarray | None, diagonal_value: float = np.inf) -> np.ndarray:
    """Compute Euclidean distances with optional previous-data rows appended."""
    matrix = compute_distance_matrix(values, history_values=history, diagonal_value=diagonal_value, return_sqrt=True)
    if history is not None:
        history_size = len(history)
        matrix[-history_size:, -history_size:] = np.max(matrix)
    np.fill_diagonal(matrix, diagonal_value)
    return matrix


def _update_pareto_front(
    new_design_x: np.ndarray,
    new_design_y: np.ndarray,
    new_point: np.ndarray,
    current_design_x: np.ndarray,
    current_design_y: np.ndarray,
    current_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert a new IRSF design into a Pareto front if it is non-dominated."""
    greater_x = new_point[0] > current_front[:, 0]
    greater_y = new_point[1] > current_front[:, 1]
    greater_equal_x = new_point[0] >= current_front[:, 0]
    greater_equal_y = new_point[1] >= current_front[:, 1]
    lower_x = np.logical_not(greater_equal_x)
    lower_y = np.logical_not(greater_equal_y)
    lower_equal_x = np.logical_not(greater_x)
    lower_equal_y = np.logical_not(greater_y)
    equal_x = new_point[0] == current_front[:, 0]
    equal_y = new_point[1] == current_front[:, 1]

    keep_current = (np.multiply(greater_x, greater_equal_y) + np.multiply(greater_y, greater_equal_x)) == 0
    keep_current = keep_current.flatten()
    dominated = np.sum(np.multiply(lower_x, lower_equal_y) + np.multiply(lower_y, lower_equal_x) + np.multiply(equal_x, equal_y))

    if np.any(keep_current):
        rows_per_design = len(new_design_x)
        front_indices = np.where(keep_current)[0]
        design_indices = np.array([index * rows_per_design + np.arange(rows_per_design) for index in front_indices]).flatten()
        next_front = current_front[front_indices]
        next_design_x = current_design_x[design_indices]
        next_design_y = current_design_y[design_indices]
    else:
        next_front = np.empty((0, len(new_point)))
        next_design_x = np.empty((0, np.shape(new_design_x)[1]))
        next_design_y = np.empty((0, np.shape(new_design_y)[1]))

    if dominated == 0:
        next_front = np.append(next_front, [new_point], axis=0)
        next_design_x = np.append(next_design_x, new_design_x, axis=0)
        next_design_y = np.append(next_design_y, new_design_y, axis=0)

    return next_design_x, next_design_y, next_front


def _combine_pareto_front(
    new_front: list[np.ndarray],
    current_front: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two Pareto-front representations into one non-dominated set."""
    if current_front is None:
        return new_front
    combined = current_front
    n_designs = len(new_front[2])
    design_size = int(len(new_front[0]) / n_designs)
    for design_index in range(n_designs):
        combined = _update_pareto_front(
            new_front[0][design_index * design_size : (design_index + 1) * design_size, :],
            new_front[1][design_index * design_size : (design_index + 1) * design_size, :],
            new_front[2][design_index, :],
            combined[0],
            combined[1],
            combined[2],
        )
    return combined


def _criterion_x(
    candidates: np.ndarray,
    *,
    max_iterations: int,
    num_restarts: int,
    design_size: int,
    history: np.ndarray | None,
    rng: np.random.Generator,
) -> float:
    """Find the best pure-space-filling minimum distance for one space."""
    best_minimum = 0.0
    best_ties = 0
    for _ in range(num_restarts):
        selected = rng.choice(len(candidates), design_size, replace=False)
        design = candidates[selected]
        distance_matrix = _compute_distance(design, history)
        minimum, points, ties = minimum_distance_summary(distance_matrix)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            design, minimum, points, ties, distance_matrix, _added, _removed, improved = _update_min_distance(
                design,
                candidates,
                minimum,
                points,
                ties,
                distance_matrix,
                history=history,
                rng=rng,
            )
            iteration += 1
        if (minimum > best_minimum) or (minimum == best_minimum and (best_ties == 0 or ties < best_ties)):
            best_minimum = float(minimum)
            best_ties = int(ties)
    return best_minimum


def _update_min_distance(
    current_design: np.ndarray,
    candidates: np.ndarray,
    current_minimum: float,
    current_points: np.ndarray,
    current_ties: int,
    distance_matrix: np.ndarray,
    *,
    history: np.ndarray | None,
    rng: np.random.Generator,
    diagonal_value: float = np.inf,
) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray, int | None, int | None, bool]:
    """Attempt a single improving swap for a maximin design state.

    Returns:
        Updated design state, criterion summaries, and a flag indicating whether
        an improving swap was found.
    """
    def refresh(row: np.ndarray, design: np.ndarray, matrix: np.ndarray, position: int) -> np.ndarray:
        """Refresh one row and column of the Euclidean distance matrix."""
        inputs = design if history is None else np.concatenate((design, history))
        values = np.sqrt(np.sum(np.square(row - inputs), axis=1))
        values[position] = diagonal_value
        updated = np.array(matrix, copy=True)
        updated[position, :] = values
        updated[:, position] = values
        return updated

    def try_swap(minimum_position: int, candidate_position: int) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray, int, int]:
        """Evaluate the effect of swapping one design row with one candidate."""
        design = np.array(current_design, copy=True)
        candidate_row = candidates[candidate_position]
        replace_at = candidate_points[minimum_position]
        design[replace_at] = candidate_row
        updated = refresh(candidate_row, design, distance_matrix, replace_at)
        minimum, points, ties = minimum_distance_summary(updated)
        return design, minimum, points, ties, updated, candidate_position, replace_at

    candidate_points = current_points[current_points < len(current_design)]
    n_points = len(candidate_points)
    n_candidates = len(candidates)
    improvements = np.zeros((n_points, n_candidates))
    ties = np.zeros((n_points, n_candidates))
    for minimum_position in range(n_points):
        for candidate_position in range(n_candidates):
            _, improvements[minimum_position, candidate_position], _, ties[minimum_position, candidate_position], _, _, _ = try_swap(
                minimum_position, candidate_position
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


def _xy_min_distance(
    design_x: np.ndarray,
    design_y: np.ndarray,
    best_x: float,
    best_y: float,
    weight: float,
    history_x: np.ndarray | None,
    history_y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int]:
    """Compute weighted and marginal distance summaries for one IRSF design."""
    distance_x = _compute_distance(design_x, history_x)
    distance_y = _compute_distance(design_y, history_y)
    if history_x is not None and history_y is not None:
        history_size = len(history_x)
        distance_x[-history_size:, -history_size:] = np.max(distance_x)
        distance_y[-history_size:, -history_size:] = np.max(distance_y)
    distance_xy = (weight / best_x) * distance_x + ((1.0 - weight) / best_y) * distance_y
    if history_x is not None and history_y is not None:
        history_size = len(history_x)
        distance_xy[-history_size:, -history_size:] = np.max(distance_xy)
    minimum, points, ties = minimum_distance_summary(distance_xy)
    return distance_xy, distance_x, distance_y, minimum, points, ties


def _update_min_xy_distance(
    design_x: np.ndarray,
    design_y: np.ndarray,
    candidates_x: np.ndarray,
    candidates_y: np.ndarray,
    current_minimum: float,
    current_points: np.ndarray,
    current_ties: int,
    distance_xy: np.ndarray,
    distance_x: np.ndarray,
    distance_y: np.ndarray,
    best_x: float,
    best_y: float,
    weight: float,
    pareto_design_x: np.ndarray,
    pareto_design_y: np.ndarray,
    pareto_front: np.ndarray,
    *,
    history_x: np.ndarray | None,
    history_y: np.ndarray | None,
    rng: np.random.Generator,
    diagonal_value: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None, int | None, bool]:
    """Attempt a single improving swap for a weighted IRSF objective.

    Returns:
        Updated design state, distance matrices, Pareto front state, swap
        bookkeeping, and a flag indicating whether an improving swap was found.
    """
    def refresh(
        row_x: np.ndarray,
        row_y: np.ndarray,
        next_design_x: np.ndarray,
        next_design_y: np.ndarray,
        replace_at: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refresh weighted and marginal distance matrices after one swap."""
        inputs = next_design_x if history_x is None else np.concatenate((next_design_x, history_x))
        distance_x_values = np.sqrt(np.sum(np.square(row_x - inputs), axis=1))
        distance_x_values[replace_at] = diagonal_value
        updated_x = np.array(distance_x, copy=True)
        updated_x[replace_at, :] = distance_x_values
        updated_x[:, replace_at] = distance_x_values

        responses = next_design_y if history_y is None else np.concatenate((next_design_y, history_y))
        distance_y_values = np.sqrt(np.sum(np.square(row_y - responses), axis=1))
        distance_y_values[replace_at] = diagonal_value
        updated_y = np.array(distance_y, copy=True)
        updated_y[replace_at, :] = distance_y_values
        updated_y[:, replace_at] = distance_y_values

        updated_xy = np.array(distance_xy, copy=True)
        updated_xy[replace_at, :] = (weight / best_x) * distance_x_values + ((1.0 - weight) / best_y) * distance_y_values
        updated_xy[:, replace_at] = updated_xy[replace_at, :]
        return updated_xy, updated_x, updated_y

    def try_swap(minimum_position: int, candidate_position: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Evaluate the effect of swapping one IRSF design row with one candidate."""
        next_design_x = np.array(design_x, copy=True)
        next_design_y = np.array(design_y, copy=True)
        row_x = candidates_x[candidate_position]
        row_y = candidates_y[candidate_position]
        replace_at = candidate_points[minimum_position]
        next_design_x[replace_at] = row_x
        next_design_y[replace_at] = row_y
        updated_xy, updated_x, updated_y = refresh(row_x, row_y, next_design_x, next_design_y, replace_at)
        minimum, points, ties = minimum_distance_summary(updated_xy)
        return next_design_x, next_design_y, minimum, points, ties, updated_xy, updated_x, updated_y, candidate_position, replace_at

    candidate_points = current_points[current_points < len(design_x)]
    n_points = len(candidate_points)
    n_candidates = len(candidates_x)
    improvements = np.zeros((n_points, n_candidates))
    ties = np.zeros((n_points, n_candidates))
    for minimum_position, candidate_position in itertools.product(range(n_points), range(n_candidates)):
        next_design_x, next_design_y, improvements[minimum_position, candidate_position], _, ties[minimum_position, candidate_position], _, updated_x, updated_y, _, _ = try_swap(
            minimum_position, candidate_position
        )
        new_point = np.array([np.min(updated_x), np.min(updated_y)])
        pareto_design_x, pareto_design_y, pareto_front = _update_pareto_front(
            next_design_x,
            next_design_y,
            new_point,
            pareto_design_x,
            pareto_design_y,
            pareto_front,
        )
    best_value = np.max(improvements)
    best_pairs = np.argwhere(improvements == best_value)
    if best_value > current_minimum:
        pair = best_pairs[int(rng.integers(0, len(best_pairs)))]
        (
            next_design_x,
            next_design_y,
            next_minimum,
            next_points,
            next_ties,
            next_distance_xy,
            next_distance_x,
            next_distance_y,
            added,
            removed,
        ) = try_swap(int(pair[0]), int(pair[1]))
        return (
            next_design_x,
            next_design_y,
            next_minimum,
            next_points,
            next_ties,
            next_distance_xy,
            next_distance_x,
            next_distance_y,
            pareto_design_x,
            pareto_design_y,
            pareto_front,
            added,
            removed,
            True,
        )
    if best_value == current_minimum:
        better_ties = np.argwhere(ties[best_pairs[:, 0], best_pairs[:, 1]] < current_ties).flatten()
        if len(better_ties) > 0:
            pair = best_pairs[int(rng.choice(better_ties))]
            (
                next_design_x,
                next_design_y,
                next_minimum,
                next_points,
                next_ties,
                next_distance_xy,
                next_distance_x,
                next_distance_y,
                added,
                removed,
            ) = try_swap(int(pair[0]), int(pair[1]))
            return (
                next_design_x,
                next_design_y,
                next_minimum,
                next_points,
                next_ties,
                next_distance_xy,
                next_distance_x,
                next_distance_y,
                pareto_design_x,
                pareto_design_y,
                pareto_front,
                added,
                removed,
                True,
            )
    return (
        design_x,
        design_y,
        current_minimum,
        candidate_points,
        current_ties,
        distance_xy,
        distance_x,
        distance_y,
        pareto_design_x,
        pareto_design_y,
        pareto_front,
        None,
        None,
        False,
    )


def _irsf_restart(
    candidates_x: np.ndarray,
    candidates_y: np.ndarray,
    *,
    best_x: float,
    best_y: float,
    weight: float,
    max_iterations: int,
    design_size: int,
    history_x: np.ndarray | None,
    history_y: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one randomized IRSF search for a single input-response weight."""
    selected = rng.choice(len(candidates_x), design_size, replace=False)
    design_x = candidates_x[selected]
    design_y = candidates_y[selected]
    distance_xy, distance_x, distance_y, minimum, points, ties = _xy_min_distance(
        design_x,
        design_y,
        best_x,
        best_y,
        weight,
        history_x,
        history_y,
    )
    pareto_design_x = design_x
    pareto_design_y = design_y
    pareto_front = np.array([[np.min(distance_x), np.min(distance_y)]])
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        (
            design_x,
            design_y,
            minimum,
            points,
            ties,
            distance_xy,
            distance_x,
            distance_y,
            pareto_design_x,
            pareto_design_y,
            pareto_front,
            _added,
            _removed,
            improved,
        ) = _update_min_xy_distance(
            design_x,
            design_y,
            candidates_x,
            candidates_y,
            minimum,
            points,
            ties,
            distance_xy,
            distance_x,
            distance_y,
            best_x,
            best_y,
            weight,
            pareto_design_x,
            pareto_design_y,
            pareto_front,
            history_x=history_x,
            history_y=history_y,
            rng=rng,
        )
        iteration += 1
    return (
        design_x,
        design_y,
        minimum,
        points,
        ties,
        distance_xy,
        distance_x,
        distance_y,
        pareto_design_x,
        pareto_design_y,
        pareto_front,
    )


def _criterion_irsf(
    candidates_x: np.ndarray,
    candidates_y: np.ndarray,
    *,
    best_x: float,
    best_y: float,
    weight: float,
    max_iterations: int,
    num_restarts: int,
    design_size: int,
    history_x: np.ndarray | None,
    history_y: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect Pareto-optimal IRSF designs for one input-response weight."""
    (
        _design_x,
        _design_y,
        minimum,
        _points,
        ties,
        _distance_xy,
        _distance_x,
        _distance_y,
        pareto_design_x,
        pareto_design_y,
        pareto_front,
    ) = _irsf_restart(
        candidates_x,
        candidates_y,
        best_x=best_x,
        best_y=best_y,
        weight=weight,
        max_iterations=max_iterations,
        design_size=design_size,
        history_x=history_x,
        history_y=history_y,
        rng=rng,
    )

    best_minimum: float | None = None
    for _ in range(num_restarts - 1):
        (
            _design_x,
            _design_y,
            minimum_candidate,
            _points,
            ties_candidate,
            _distance_xy,
            _distance_x,
            _distance_y,
            pareto_design_x_candidate,
            pareto_design_y_candidate,
            pareto_front_candidate,
        ) = _irsf_restart(
            candidates_x,
            candidates_y,
            best_x=best_x,
            best_y=best_y,
            weight=weight,
            max_iterations=max_iterations,
            design_size=design_size,
            history_x=history_x,
            history_y=history_y,
            rng=rng,
        )
        if best_minimum is None or (minimum_candidate > best_minimum or (minimum_candidate == best_minimum and ties_candidate < ties)):
            best_minimum = float(minimum_candidate)
            ties = ties_candidate
        for design_index in range(pareto_front_candidate.shape[0]):
            pareto_design_x, pareto_design_y, pareto_front = _update_pareto_front(
                pareto_design_x_candidate[design_index * design_size : (design_index + 1) * design_size, :],
                pareto_design_y_candidate[design_index * design_size : (design_index + 1) * design_size, :],
                pareto_front_candidate[design_index, :],
                pareto_design_x,
                pareto_design_y,
                pareto_front,
            )
    return pareto_design_x, pareto_design_y, pareto_front


def design_input_response(
    *,
    setup: DesignSetup,
    design_size: int,
    num_restarts: int,
    random_state: int | np.random.Generator | None = None,
    max_iterations: int = 1000,
    weight_grid: np.ndarray | None = None,
) -> InputResponseDesignResult:
    """Construct an input-response space-filling Pareto front.

    Args:
        setup: Validated design setup with at least one response column.
        design_size: Number of rows per design.
        num_restarts: Number of random restarts used for each search.
        random_state: Optional seed or generator.
        max_iterations: Maximum local-improvement iterations per restart.
        weight_grid: Optional grid of input-response tradeoff weights.

    Returns:
        IRSF result containing the Pareto front and one design table per
        non-dominated solution.
    """
    roles = setup.roles
    if not roles.responses:
        raise ConfigurationError("At least one response column is required for input-response design.")

    candidate = setup.candidate
    previous = setup.previous
    candidate_x = scale_columns(candidate[roles.inputs], columns=roles.inputs, bounds=setup.bounds).to_numpy()
    candidate_y = scale_columns(candidate[roles.responses], columns=roles.responses, bounds=setup.bounds).to_numpy()

    history_x = None
    history_y = None
    if previous is not None:
        history_x = scale_columns(previous[roles.inputs], columns=roles.inputs, bounds=setup.bounds).to_numpy()
        history_y = scale_columns(previous[roles.responses], columns=roles.responses, bounds=setup.bounds).to_numpy()

    rng = _generator(random_state)
    start = time.time()
    best_x = _criterion_x(
        candidate_x,
        max_iterations=max_iterations,
        num_restarts=num_restarts,
        design_size=design_size,
        history=history_x,
        rng=rng,
    )
    best_y = _criterion_x(
        candidate_y,
        max_iterations=max_iterations,
        num_restarts=num_restarts,
        design_size=design_size,
        history=history_y,
        rng=rng,
    )

    weight_values = weight_grid if weight_grid is not None else np.linspace(0.1, 0.9, 5)
    pareto_x: dict[int, np.ndarray] = {}
    pareto_y: dict[int, np.ndarray] = {}
    pareto_values: dict[int, np.ndarray] = {}

    for index, weight in enumerate(weight_values):
        pareto_x[index], pareto_y[index], pareto_values[index] = _criterion_irsf(
            candidate_x,
            candidate_y,
            best_x=best_x,
            best_y=best_y,
            weight=float(weight),
            max_iterations=max_iterations,
            num_restarts=num_restarts,
            design_size=design_size,
            history_x=history_x,
            history_y=history_y,
            rng=rng,
        )
        pareto_x[index] = inverse_scale_columns(pareto_x[index], columns=roles.inputs, bounds=setup.bounds)
        pareto_y[index] = inverse_scale_columns(pareto_y[index], columns=roles.responses, bounds=setup.bounds)

    combined: tuple[np.ndarray, np.ndarray, np.ndarray] | list[np.ndarray] | None = None
    for index in range(len(weight_values)):
        combined = _combine_pareto_front([pareto_x[index], pareto_y[index], pareto_values[index]], combined)  # type: ignore[arg-type]

    if combined is None:
        raise RuntimeError("Input-response search did not produce a result.")

    combined_x, combined_y, combined_front = combined  # type: ignore[misc]
    sort_index = np.argsort(combined_front, axis=0)[:, 0]
    pareto_front = combined_front[sort_index]
    pareto_frame = pd.DataFrame(pareto_front, columns=["Best Input", "Best Response"])
    pareto_frame.insert(0, "Design", pareto_frame.index + 1)

    designs: dict[int, pd.DataFrame] = {}
    for output_index, combined_index in enumerate(sort_index, start=1):
        design_x = combined_x[(combined_index * design_size) + np.arange(design_size), :]
        design_y = combined_y[(combined_index * design_size) + np.arange(design_size), :]
        designs[output_index] = pd.DataFrame(
            np.concatenate((design_x, design_y), axis=1),
            columns=roles.inputs + roles.responses,
        )

    _elapsed = time.time() - start
    return InputResponseDesignResult(
        pareto_front=pareto_frame,
        designs=designs,
        mode="maximin",
        design_size=design_size,
        num_restarts=num_restarts,
    )


def estimate_input_response_runtime(
    *,
    setup: DesignSetup,
    design_size: int,
    target_restarts: int,
    calibration_restarts: int = 5,
    random_state: int | np.random.Generator | None = None,
    max_iterations: int = 1000,
    weight_grid: np.ndarray | None = None,
) -> dict[str, object]:
    """Estimate runtime for an IRSF search.

    Args:
        setup: Validated design setup.
        design_size: Number of rows per design.
        target_restarts: Restart count to estimate.
        calibration_restarts: Smaller restart count used to calibrate the
            estimate.
        random_state: Optional seed or generator.
        max_iterations: Maximum local-improvement iterations per restart.
        weight_grid: Optional grid of input-response tradeoff weights.

    Returns:
        Dictionary with the projected runtime in seconds.
    """
    start = time.time()
    _ = design_input_response(
        setup=setup,
        design_size=design_size,
        num_restarts=calibration_restarts,
        random_state=random_state,
        max_iterations=max_iterations,
        weight_grid=weight_grid,
    )
    elapsed = time.time() - start
    estimated = elapsed * (target_restarts / calibration_restarts)
    return {
        "mode": "maximin",
        "design_size": design_size,
        "target_restarts": target_restarts,
        "calibration_restarts": calibration_restarts,
        "estimated_seconds": estimated,
    }
