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


class _MarsModel:
    """Multivariate Adaptive Regression Splines (Friedman, 1991).

    Native NumPy implementation of the MARS method. A forward pass greedily adds
    pairs of hinge basis functions ``max(0, +/-(x - knot))``; a backward pass
    then removes terms to minimize the generalized cross-validation (GCV) score.
    Defaults follow common MARS settings: up to ``max_terms`` basis functions
    and interaction degree ``min(n_inputs, 8)``. No external dependency is
    required.
    """

    def __init__(self, max_terms: int = 100, max_degree: int | None = None, penalty: float = 3.0) -> None:
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self._terms: list[list[tuple[int, float, int]]] | None = None
        self._coefficients: np.ndarray | None = None

    @staticmethod
    def _evaluate_term(term: list[tuple[int, float, int]], x: np.ndarray) -> np.ndarray:
        """Evaluate one basis term (a product of hinge factors) over rows of ``x``."""
        result = np.ones(x.shape[0], dtype=float)
        for feature, knot, sign in term:
            result = result * np.maximum(0.0, sign * (x[:, feature] - knot))
        return result

    def _design(self, terms: list[list[tuple[int, float, int]]], x: np.ndarray) -> np.ndarray:
        """Build the design matrix whose columns are the evaluated terms."""
        return np.column_stack([self._evaluate_term(term, x) for term in terms])

    @staticmethod
    def _fit_least_squares(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Least-squares fit; return coefficients and residual sum of squares."""
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        return coefficients, float(residual @ residual)

    def _gcv(self, rss: float, n_terms: int, n_samples: int) -> float:
        """Generalized cross-validation score for a model with ``n_terms`` terms."""
        cost = n_terms + self.penalty * (n_terms - 1) / 2.0
        denominator = 1.0 - cost / n_samples
        if denominator <= 0.0:
            return float("inf")
        return (rss / n_samples) / (denominator * denominator)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_MarsModel":
        """Fit a MARS model by forward selection followed by GCV backward pruning."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = x.shape
        max_degree = self.max_degree if self.max_degree is not None else min(n_features, 8)
        max_terms = min(self.max_terms, max(1, n_samples - 1))

        terms: list[list[tuple[int, float, int]]] = [[]]  # intercept
        _, best_rss = self._fit_least_squares(self._design(terms, x), y)

        # Forward pass: greedily add the best hinge pair until no improvement.
        while len(terms) + 2 <= max_terms:
            best_choice = None
            best_choice_rss = best_rss
            for parent in terms:
                if len(parent) >= max_degree:
                    continue
                used_features = {factor[0] for factor in parent}
                parent_active = self._evaluate_term(parent, x) > 0
                for feature in range(n_features):
                    if feature in used_features:
                        continue
                    for knot in np.unique(x[parent_active, feature]):
                        trial = terms + [
                            parent + [(feature, float(knot), 1)],
                            parent + [(feature, float(knot), -1)],
                        ]
                        _, rss = self._fit_least_squares(self._design(trial, x), y)
                        if rss < best_choice_rss - 1e-12:
                            best_choice_rss = rss
                            best_choice = (parent, feature, float(knot))
            if best_choice is None:
                break
            parent, feature, knot = best_choice
            terms = terms + [
                parent + [(feature, knot, 1)],
                parent + [(feature, knot, -1)],
            ]
            best_rss = best_choice_rss

        # Backward pass: drop terms one at a time, keeping the lowest-GCV subset.
        _, rss = self._fit_least_squares(self._design(terms, x), y)
        best_terms = list(terms)
        best_gcv = self._gcv(rss, len(terms), n_samples)
        current = list(terms)
        while len(current) > 1:
            drop_index = None
            drop_rss = float("inf")
            for index in range(1, len(current)):  # never drop the intercept at index 0
                trial = current[:index] + current[index + 1 :]
                _, rss = self._fit_least_squares(self._design(trial, x), y)
                if rss < drop_rss:
                    drop_rss = rss
                    drop_index = index
            current = current[:drop_index] + current[drop_index + 1 :]
            gcv = self._gcv(drop_rss, len(current), n_samples)
            if gcv < best_gcv:
                best_gcv = gcv
                best_terms = list(current)

        self._terms = best_terms
        self._coefficients, _ = self._fit_least_squares(self._design(best_terms, x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict responses from the fitted MARS basis expansion."""
        if self._terms is None or self._coefficients is None:
            raise RuntimeError("Model has not been fit.")
        x = np.asarray(x, dtype=float)
        return np.asarray(self._design(self._terms, x) @ self._coefficients).reshape(-1)


def _surface_model(method: str):
    """Build the response-surface model requested by name."""
    if method == "linear":
        return _LeastSquaresModel(degree=1)
    if method == "quadratic":
        return _LeastSquaresModel(degree=2)
    if method == "cubic":
        return _LeastSquaresModel(degree=3)
    if method == "mars":
        return _MarsModel()
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
        method: Response-surface method: one of ``"linear"``, ``"quadratic"``,
            ``"cubic"``, or ``"mars"`` (MARS is built in; no extra install
            needed).
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
        method: Response-surface method: one of ``"linear"``, ``"quadratic"``,
            ``"cubic"``, or ``"mars"`` (MARS is built in; no extra install
            needed).
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
