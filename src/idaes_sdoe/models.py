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
"""Core dataclasses used to move SDoE setup and result state through the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Role = Literal["index", "input", "response", "weight"]


@dataclass(frozen=True)
class ColumnRoles:
    """Describe how table columns participate in an SDoE workflow.

    Attributes:
        inputs: Columns used to define the design space.
        index: Optional identifier column for tracking selected rows.
        responses: Optional response columns used by IRSF workflows.
        weight: Optional weight column used by NUSF workflows.
    """

    inputs: list[str]
    index: str | None = None
    responses: list[str] = field(default_factory=list)
    weight: str | None = None

    def included_columns(self) -> list[str]:
        """Return the columns actively used by a design workflow.

        Returns:
            Ordered list of index, input, response, and weight columns that are
            part of the current setup.
        """
        columns: list[str] = []
        if self.index:
            columns.append(self.index)
        columns.extend(self.inputs)
        columns.extend(self.responses)
        if self.weight:
            columns.append(self.weight)
        return columns


@dataclass(frozen=True)
class DesignSetup:
    """Validated setup state shared by the SDoE design algorithms.

    Attributes:
        candidate: Candidate table from which new design rows may be selected.
        previous: Optional previous-data table representing runs already
            collected.
        roles: Column-role mapping for the current workflow.
        bounds: Per-column min/max bounds used for scaling and distance
            calculations.
    """

    candidate: pd.DataFrame
    previous: pd.DataFrame | None
    roles: ColumnRoles
    bounds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class UniformDesignResult:
    """Result of a uniform space-filling search.

    Attributes:
        design: Selected design rows.
        selected_indices: Candidate-row identifiers selected into the design.
        criterion_value: Final minimax or maximin criterion value.
        distance_matrix: Distance matrix for the selected design.
        mode: Uniform criterion used, typically ``"minimax"`` or
            ``"maximin"``.
        design_size: Number of rows in the design.
        num_restarts: Number of random restarts evaluated.
        elapsed_time: Runtime in seconds.
    """

    design: pd.DataFrame
    selected_indices: list[Any]
    criterion_value: float
    distance_matrix: Any
    mode: str
    design_size: int
    num_restarts: int
    elapsed_time: float


@dataclass(frozen=True)
class NonUniformDesignResult:
    """Result of a non-uniform space-filling search.

    Attributes:
        design: Selected design rows in original units.
        scaled_design: Selected design rows after input and weight scaling.
        selected_indices: Candidate-row identifiers selected into the design.
        criterion_value: Final weighted maximin criterion value.
        distance_matrix: Weighted distance matrix for the selected design.
        mode: Design criterion used by the search.
        design_size: Number of rows in the design.
        num_restarts: Number of random restarts evaluated.
        mwr: Maximum weight ratio used in the weight scaling.
        elapsed_time: Runtime in seconds.
        tie_count: Number of tied minimum-distance pairs in the winning design.
    """

    design: pd.DataFrame
    scaled_design: pd.DataFrame
    selected_indices: list[Any]
    criterion_value: float
    distance_matrix: Any
    mode: str
    design_size: int
    num_restarts: int
    mwr: int
    elapsed_time: float
    tie_count: int


@dataclass(frozen=True)
class InputResponseDesignResult:
    """Result of an input-response space-filling search.

    Attributes:
        pareto_front: Table of non-dominated designs in input-response
            criterion space.
        designs: Mapping from design id to the corresponding selected design
            table.
        mode: Design criterion used by the search.
        design_size: Number of rows per design.
        num_restarts: Number of random restarts evaluated for each IRSF search.
    """

    pareto_front: pd.DataFrame
    designs: dict[int, pd.DataFrame]
    mode: str
    design_size: int
    num_restarts: int

    @property
    def num_designs(self) -> int:
        """Return the number of Pareto-optimal designs in the result."""
        return len(self.designs)


@dataclass(frozen=True)
class RunOrderResult:
    """Result of reordering a design for execution.

    Attributes:
        design: Original design table.
        ordered_design: Reordered design table.
        permutation: Row-order permutation applied to the original design.
        path_length: Total path length through the ordered scaled inputs.
        method: Ordering method used to compute the permutation.
    """

    design: pd.DataFrame
    ordered_design: pd.DataFrame
    permutation: list[int]
    path_length: float
    method: str


@dataclass(frozen=True)
class InputSpec:
    """Specification for one input dimension during candidate generation.

    Attributes:
        name: Input name.
        lower: Lower bound for generated values.
        upper: Upper bound for generated values.
        default: Optional fixed/default value used when the input does not vary.
        variable: Whether the input should be sampled or held fixed.
        distribution: Distribution name used to transform unit samples.
        parameters: Distribution-specific parameters.
    """

    name: str
    lower: float
    upper: float
    default: float | None = None
    variable: bool = True
    distribution: str = "uniform"
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Result of a candidate-generation workflow.

    Attributes:
        samples: Generated candidate table.
        scheme: Sampling scheme used to generate the table.
        num_samples: Number of generated rows.
    """

    samples: pd.DataFrame
    scheme: str
    num_samples: int


@dataclass(frozen=True)
class ResponseSurfaceValidation:
    """Validation summary for a fitted response-surface model.

    Attributes:
        method: Response-surface method name.
        observed: Observed target values used for validation.
        predicted: Predicted values aligned with ``observed``.
        rmse: Root-mean-square validation error.
        r2: Coefficient of determination for the validation predictions.
        model: Fitted model object used for prediction or imputation.
    """

    method: str
    observed: pd.Series
    predicted: pd.Series
    rmse: float
    r2: float
    model: Any
