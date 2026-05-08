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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy import stats
from scipy.stats import qmc

from ..exceptions import ConfigurationError
from ..models import CandidateGenerationResult, InputSpec


def load_template_specs(path: str | Path) -> list[InputSpec]:
    """Load first-experiment input specifications from a template CSV.

    Args:
        path: Template file containing at least min and max rows.

    Returns:
        Ordered list of input specifications derived from the template.
    """
    template = pd.read_csv(path)
    if len(template) < 2:
        raise ConfigurationError("Template files must include at least min and max rows.")
    headers = template.columns.tolist()
    min_row = template.iloc[0]
    max_row = template.iloc[1]
    default_row = template.iloc[2] if len(template) > 2 else None
    specs: list[InputSpec] = []
    for header in headers:
        default = None if default_row is None else float(default_row[header])
        specs.append(
            InputSpec(
                name=header,
                lower=float(min_row[header]),
                upper=float(max_row[header]),
                default=default,
            )
        )
    return specs


def specs_from_previous(frame: pd.DataFrame, *, columns: list[str] | None = None) -> list[InputSpec]:
    """Infer input specifications from an existing data table.

    Args:
        frame: Source table used to infer bounds and defaults.
        columns: Optional subset of columns to convert.

    Returns:
        Ordered list of input specifications.
    """
    use_columns = columns or frame.columns.tolist()
    specs: list[InputSpec] = []
    for column in use_columns:
        specs.append(
            InputSpec(
                name=column,
                lower=float(frame[column].min()),
                upper=float(frame[column].max()),
                default=float(frame[column].mean()),
            )
        )
    return specs


def _distribution(spec: InputSpec):
    """Build a SciPy distribution object for one input specification."""
    parameters = spec.parameters
    if spec.distribution == "uniform":
        return stats.uniform(loc=spec.lower, scale=spec.upper - spec.lower)
    if spec.distribution == "normal":
        return stats.norm(loc=parameters.get("mean", spec.default or (spec.lower + spec.upper) / 2), scale=parameters["sd"])
    if spec.distribution == "lognormal":
        return stats.lognorm(s=parameters["sigma"], scale=np.exp(parameters["mean"]))
    if spec.distribution == "triangular":
        peak = parameters["mode"]
        c = (peak - spec.lower) / (spec.upper - spec.lower)
        return stats.triang(c=c, loc=spec.lower, scale=spec.upper - spec.lower)
    if spec.distribution == "beta":
        return stats.beta(a=parameters["a"], b=parameters["b"], loc=spec.lower, scale=spec.upper - spec.lower)
    if spec.distribution == "gamma":
        return stats.gamma(a=parameters["shape"], scale=parameters["scale"])
    if spec.distribution == "exponential":
        return stats.expon(scale=parameters["scale"])
    raise ConfigurationError(f"Unsupported distribution '{spec.distribution}'.")


def _unit_samples(
    dimension: int,
    num_samples: int,
    *,
    scheme: str,
    random_state: int | None,
) -> np.ndarray:
    """Generate samples on the unit hypercube for a chosen sampling scheme.

    Args:
        dimension: Number of variable inputs.
        num_samples: Number of rows to generate.
        scheme: Sampling scheme name.
        random_state: Optional random seed.

    Returns:
        ``num_samples x dimension`` array of unit-hypercube samples.
    """
    if scheme == "monte_carlo":
        rng = np.random.default_rng(random_state)
        return rng.random((num_samples, dimension))
    if scheme in {"quasi_monte_carlo", "sobol"}:
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=random_state)
        return sampler.random(num_samples)
    if scheme == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=dimension, seed=random_state)
        return sampler.random(num_samples)
    if scheme == "orthogonal_array":
        sampler = qmc.LatinHypercube(d=dimension, strength=2, seed=random_state)
        return sampler.random(num_samples)
    if scheme == "metis":
        rng = np.random.default_rng(random_state)
        pool_size = max(1024, num_samples * 16)
        pool = rng.random((pool_size, dimension))
        centers, _labels = kmeans2(pool, num_samples, minit="points")
        return np.clip(centers, 0.0, 1.0)
    raise ConfigurationError(f"Unsupported sampling scheme '{scheme}'.")


def generate_candidates(
    specs: list[InputSpec],
    *,
    num_samples: int,
    scheme: str = "latin_hypercube",
    random_state: int | None = None,
) -> CandidateGenerationResult:
    """Generate a candidate table from input specifications.

    Args:
        specs: Input specifications describing bounds, defaults, and optional
            distributions.
        num_samples: Number of candidate rows to generate.
        scheme: Sampling scheme used on the unit hypercube.
        random_state: Optional random seed.

    Returns:
        Candidate-generation result with the sampled table and run metadata.
    """
    variable_specs = [spec for spec in specs if spec.variable]
    fixed_specs = [spec for spec in specs if not spec.variable]
    units = (
        _unit_samples(len(variable_specs), num_samples, scheme=scheme, random_state=random_state)
        if variable_specs
        else np.empty((num_samples, 0))
    )
    units = np.clip(units, 1e-9, 1.0 - 1e-9)

    data: dict[str, np.ndarray] = {}
    for column_index, spec in enumerate(variable_specs):
        distribution = _distribution(spec)
        values = distribution.ppf(units[:, column_index])
        values = np.clip(values, spec.lower, spec.upper)
        data[spec.name] = values
    for spec in fixed_specs:
        fill_value = spec.default if spec.default is not None else spec.lower
        data[spec.name] = np.full(num_samples, fill_value, dtype=float)

    ordered = {spec.name: data[spec.name] for spec in specs}
    return CandidateGenerationResult(
        samples=pd.DataFrame(ordered),
        scheme=scheme,
        num_samples=num_samples,
    )
