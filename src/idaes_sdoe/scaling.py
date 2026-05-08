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

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .exceptions import ConfigurationError


def build_bounds(
    candidate: pd.DataFrame,
    previous: pd.DataFrame | None,
    columns: Iterable[str],
    overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Build min/max bounds for a set of columns.

    Args:
        candidate: Candidate table.
        previous: Optional previous-data table included when inferring bounds.
        columns: Columns to process.
        overrides: Optional explicit bounds keyed by column name.

    Returns:
        Mapping from column name to ``(lower, upper)`` bounds.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        if overrides and column in overrides:
            bounds[column] = overrides[column]
            continue
        series = candidate[column]
        if previous is not None and column in previous:
            series = pd.concat([series, previous[column]], ignore_index=True)
        lower = float(series.min())
        upper = float(series.max())
        bounds[column] = (lower, upper)
    return bounds


def scale_columns(
    values: pd.DataFrame | np.ndarray,
    *,
    columns: list[str] | None = None,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame | np.ndarray:
    """Scale tabular values into the unit interval using stored bounds.

    Args:
        values: DataFrame or array to scale.
        columns: Column names corresponding to the values being scaled.
        bounds: Per-column bounds.

    Returns:
        Scaled object of the same high-level type as ``values``.
    """
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        for column in columns or []:
            lower, upper = bounds[column]
            scale = upper - lower
            if scale == 0:
                raise ConfigurationError(f"Zero-width bounds for column '{column}'.")
            frame[column] = (frame[column] - lower) / scale
        return frame

    if columns is None:
        raise ConfigurationError("Column names are required when scaling arrays.")
    array = np.array(values, copy=True, dtype=float)
    for index, column in enumerate(columns):
        lower, upper = bounds[column]
        scale = upper - lower
        if scale == 0:
            raise ConfigurationError(f"Zero-width bounds for column '{column}'.")
        array[:, index] = (array[:, index] - lower) / scale
    return array


def inverse_scale_columns(
    values: np.ndarray,
    *,
    columns: list[str],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Map unit-scaled arrays back to their original numeric ranges.

    Args:
        values: Scaled numeric array.
        columns: Column names aligned with array positions.
        bounds: Per-column bounds used for inverse scaling.

    Returns:
        Array transformed back to original engineering units.
    """
    array = np.array(values, copy=True, dtype=float)
    for index, column in enumerate(columns):
        lower, upper = bounds[column]
        array[:, index] = array[:, index] * (upper - lower) + lower
    return array


def scale_factors_for(columns: list[str], bounds: dict[str, tuple[float, float]]) -> pd.Series:
    """Return span-based distance scale factors for selected columns.

    Args:
        columns: Columns to include.
        bounds: Per-column bounds.

    Returns:
        Series of ``upper - lower`` spans keyed by column name.
    """
    return pd.Series({column: bounds[column][1] - bounds[column][0] for column in columns})


def scale_weights(values: np.ndarray, *, method: str, mwr: int) -> np.ndarray:
    """Scale weights into the range ``[1, mwr]``.

    Args:
        values: Raw weight values.
        method: Weight-scaling method.
        mwr: Maximum weight ratio to assign after scaling.

    Returns:
        Scaled weight vector suitable for the NUSF distance criterion.
    """
    scaled = np.array(values, copy=True, dtype=float)
    if method not in {"direct_mwr", "ranked_mwr"}:
        raise ConfigurationError(f"Unknown weight scaling method '{method}'.")
    if method == "ranked_mwr":
        scaled = rankdata(scaled, method="dense")
    lower = np.min(scaled)
    upper = np.max(scaled)
    if lower == upper:
        return np.ones_like(scaled, dtype=float)
    return 1.0 + (mwr - 1.0) * (scaled - lower) / (upper - lower)
