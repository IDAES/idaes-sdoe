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

from typing import Optional

import numpy as np


def compute_distance_matrix(
    values: np.ndarray,
    *,
    scale_factors: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    history_values: Optional[np.ndarray] = None,
    history_weights: Optional[np.ndarray] = None,
    diagonal_value: float = np.inf,
    return_sqrt: bool = False,
) -> np.ndarray:
    """Compute a pairwise distance matrix for candidate or design points.

    Args:
        values: Two-dimensional array of rows to compare.
        scale_factors: Optional per-column scaling factors applied before
            distance calculation.
        weights: Optional per-row weights used by the non-uniform design
            criterion.
        history_values: Optional rows appended for distance comparisons against
            previous data.
        history_weights: Optional weights aligned with ``history_values``.
        diagonal_value: Value written to the diagonal after the matrix is
            formed.
        return_sqrt: If ``True``, return Euclidean distances instead of squared
            distances.

    Returns:
        A square distance matrix over ``values`` and any appended history rows.
    """
    matrix = np.array(values, dtype=float, copy=True)
    if history_values is not None:
        matrix = np.concatenate((matrix, history_values), axis=0)
    if matrix.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two points are required.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("All values must be finite.")

    n_rows, n_cols = matrix.shape
    if scale_factors is not None:
        scale = np.asarray(scale_factors, dtype=float)
        matrix = matrix / np.repeat(scale.reshape(1, n_cols), n_rows, axis=0)
        diagonal_value = 10.0

    deltas = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sum(np.square(deltas), axis=2)

    if weights is not None:
        scaled_weights = np.asarray(weights, dtype=float)
        if history_weights is not None:
            scaled_weights = np.concatenate((scaled_weights, history_weights), axis=0)
        distances = distances * np.outer(scaled_weights, scaled_weights)
        diagonal_value = 9999.0

    if return_sqrt:
        distances = np.sqrt(distances)

    np.fill_diagonal(distances, diagonal_value)
    return distances


def minimum_distance_summary(distance_matrix: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Summarize the minimum entry in a distance matrix.

    Args:
        distance_matrix: Pairwise distance matrix.

    Returns:
        A tuple containing the minimum value, the unique point indices involved
        in minimum-distance ties, and the number of tied upper-triangular pairs.
    """
    minimum = np.min(distance_matrix)
    points = np.argwhere(np.triu(distance_matrix) == minimum)
    tie_count = points.shape[0]
    unique_points = np.unique(points.flatten())
    return float(minimum), unique_points, int(tie_count)


def path_length(values: np.ndarray) -> float:
    """Compute the total path length through ordered points.

    Args:
        values: Ordered point coordinates.

    Returns:
        Sum of Euclidean step lengths between consecutive rows.
    """
    if len(values) < 2:
        return 0.0
    deltas = np.diff(values, axis=0)
    return float(np.sum(np.sqrt(np.sum(np.square(deltas), axis=1))))
