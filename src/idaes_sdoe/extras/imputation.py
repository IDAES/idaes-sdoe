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

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from ..exceptions import OptionalDependencyError
from ..models import ResponseSurfaceValidation


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root-mean-square prediction error."""
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute coefficient of determination for predictions."""
    mean = np.mean(observed)
    total = np.sum(np.square(observed - mean))
    if total == 0:
        return 1.0
    residual = np.sum(np.square(observed - predicted))
    return float(1.0 - (residual / total))


def _polynomial_features(values: np.ndarray, degree: int) -> np.ndarray:
    """Expand inputs into polynomial features up to a given degree."""
    n_samples, n_features = values.shape
    columns = [np.ones(n_samples)]
    for current_degree in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), current_degree):
            term = np.ones(n_samples)
            for feature_index in combo:
                term = term * values[:, feature_index]
            columns.append(term)
    return np.column_stack(columns)


@dataclass
class _LeastSquaresModel:
    degree: int
    coefficients: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_LeastSquaresModel":
        """Fit a polynomial least-squares model to observed data."""
        features = _polynomial_features(x, self.degree)
        self.coefficients, *_ = np.linalg.lstsq(features, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict responses from polynomial features."""
        features = _polynomial_features(x, self.degree)
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit.")
        return features @ self.coefficients


@dataclass
class _RbfModel:
    interpolator: RBFInterpolator | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RbfModel":
        """Fit a radial-basis interpolator to observed data."""
        self.interpolator = RBFInterpolator(x, y, kernel="gaussian")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict responses with the fitted radial-basis interpolator."""
        if self.interpolator is None:
            raise RuntimeError("Model has not been fit.")
        values = self.interpolator(x)
        return np.asarray(values).reshape(-1)


def _surface_model(method: str):
    """Build the response-surface model requested by name."""
    if method == "linear":
        return _LeastSquaresModel(degree=1)
    if method == "quadratic":
        return _LeastSquaresModel(degree=2)
    if method == "cubic":
        return _LeastSquaresModel(degree=3)
    if method == "gaussian_process":
        return _RbfModel()
    if method == "mars":
        try:
            from pyearth import Earth
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OptionalDependencyError(
                "MARS support requires the optional 'mars' extra: pip install -e \".[mars]\"."
            ) from exc

        class _EarthModel:
            def __init__(self) -> None:
                """Initialize the optional MARS estimator wrapper."""
                self.model = Earth()

            def fit(self, x: np.ndarray, y: np.ndarray) -> "_EarthModel":
                """Fit the optional MARS model."""
                self.model.fit(x, y)
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                """Predict responses with the fitted MARS model."""
                return np.asarray(self.model.predict(x)).reshape(-1)

        return _EarthModel()
    raise ValueError(f"Unsupported response-surface method '{method}'.")


def _cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    cv_splits: int,
    random_state: int | None,
) -> np.ndarray:
    """Generate cross-validated predictions for observed responses."""
    n_samples = len(y)
    if n_samples < 2:
        return _surface_model(method).fit(x, y).predict(x)

    splits = min(cv_splits, n_samples)
    if splits < 2:
        return _surface_model(method).fit(x, y).predict(x)

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, splits)
    predicted = np.zeros(n_samples, dtype=float)
    for fold in folds:
        train = np.setdiff1d(indices, fold, assume_unique=True)
        model = _surface_model(method)
        model.fit(x[train], y[train])
        predicted[fold] = model.predict(x[fold])
    return predicted


def fit_response_surface(
    frame: pd.DataFrame,
    *,
    input_columns: list[str],
    target_column: str,
    method: str,
    cv_splits: int = 5,
    random_state: int | None = None,
) -> ResponseSurfaceValidation:
    """Fit and validate a response-surface model for one target column.

    Args:
        frame: Source table containing inputs and target values.
        input_columns: Input columns used by the surface model.
        target_column: Target column to model.
        method: Response-surface method name.
        cv_splits: Number of cross-validation folds.
        random_state: Optional seed for fold shuffling.

    Returns:
        Validation report containing predictions, metrics, and the fitted model.
    """
    observed_frame = frame.dropna(subset=input_columns + [target_column]).copy()
    x = observed_frame[input_columns].to_numpy(dtype=float)
    y = observed_frame[target_column].to_numpy(dtype=float)
    predicted = _cross_validated_predictions(
        x,
        y,
        method=method,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    model = _surface_model(method).fit(x, y)
    return ResponseSurfaceValidation(
        method=method,
        observed=observed_frame[target_column],
        predicted=pd.Series(predicted, index=observed_frame.index),
        rmse=_rmse(y, predicted),
        r2=_r2(y, predicted),
        model=model,
    )


def impute_missing_values(
    frame: pd.DataFrame,
    *,
    input_columns: list[str],
    target_columns: str | Sequence[str],
    method: str,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict[str, ResponseSurfaceValidation]]:
    """Impute missing target values with fitted response surfaces.

    Args:
        frame: Source table containing missing target values.
        input_columns: Input columns supplied to each response-surface model.
        target_columns: One target column or a sequence of target columns.
        method: Response-surface method name.
        random_state: Optional seed for model validation shuffling.

    Returns:
        Tuple of the completed table and per-target validation reports.
    """
    columns = [target_columns] if isinstance(target_columns, str) else list(target_columns)
    result = frame.copy()
    reports: dict[str, ResponseSurfaceValidation] = {}
    for column in columns:
        validation = fit_response_surface(
            result,
            input_columns=input_columns,
            target_column=column,
            method=method,
            random_state=random_state,
        )
        missing = result[column].isna()
        if missing.any():
            x_missing = result.loc[missing, input_columns].to_numpy(dtype=float)
            result.loc[missing, column] = validation.model.predict(x_missing)
        reports[column] = validation
    return result, reports
