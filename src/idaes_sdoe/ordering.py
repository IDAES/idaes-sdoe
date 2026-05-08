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

try:
    from python_tsp.exact import solve_tsp_dynamic_programming
except ImportError:  # pragma: no cover - exercised by fallback path
    solve_tsp_dynamic_programming = None

from .distance import compute_distance_matrix, path_length
from .models import RunOrderResult
from .scaling import build_bounds, scale_columns


def _greedy_tsp(distance_matrix: np.ndarray) -> list[int]:
    """Build a simple nearest-neighbor tour through a distance matrix.

    Args:
        distance_matrix: Square matrix of pairwise distances.

    Returns:
        Permutation of row indices representing a greedy traversal order.
    """
    remaining = set(range(1, len(distance_matrix)))
    order = [0]
    while remaining:
        current = order[-1]
        next_node = min(remaining, key=lambda node: distance_matrix[current, node])
        order.append(next_node)
        remaining.remove(next_node)
    return order


def order_runs(
    design: pd.DataFrame,
    *,
    input_columns: list[str],
    bounds: dict[str, tuple[float, float]] | None = None,
    exact: bool = True,
) -> RunOrderResult:
    """Order design rows to reduce travel through input space.

    Args:
        design: Design table to reorder.
        input_columns: Columns used to measure movement cost.
        bounds: Optional scaling bounds. If omitted, bounds are inferred from
            the design itself.
        exact: If ``True``, use the exact TSP solver when available; otherwise
            use the greedy fallback.

    Returns:
        Run-order result containing the original design, ordered design,
        permutation, path length, and method name.
    """
    if bounds is None:
        bounds = build_bounds(design, None, input_columns)
    scaled = scale_columns(design[input_columns], columns=input_columns, bounds=bounds)
    distance_matrix = compute_distance_matrix(scaled.to_numpy(), diagonal_value=0.0, return_sqrt=True)
    if exact and solve_tsp_dynamic_programming is not None:
        permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
        method = "tsp-exact"
    else:
        permutation = _greedy_tsp(distance_matrix)
        method = "tsp-greedy"
    ordered = design.iloc[permutation].reset_index(drop=True)
    ordered_scaled = scaled.iloc[permutation].to_numpy()
    return RunOrderResult(
        design=design.copy(),
        ordered_design=ordered,
        permutation=list(permutation),
        path_length=path_length(ordered_scaled),
        method=method,
    )
