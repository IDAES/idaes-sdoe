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
import pandas as pd

from idaes_sdoe.extras.candidate_generation import generate_candidates
from idaes_sdoe.extras.imputation import fit_response_surface, impute_missing_values
from idaes_sdoe.models import InputSpec


def test_candidate_generation_smoke():
    specs = [
        InputSpec(name="X1", lower=-1.0, upper=1.0),
        InputSpec(name="X2", lower=-1.0, upper=1.0),
    ]
    result = generate_candidates(specs, num_samples=16, scheme="latin_hypercube", random_state=23)
    assert result.samples.shape == (16, 2)


def test_imputation_smoke():
    frame = pd.DataFrame(
        {
            "X1": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "X2": [1.0, 0.5, 0.0, -0.5, -1.0],
            "Y": [0.0, 0.3, None, 0.8, 1.0],
        }
    )
    validation = fit_response_surface(
        frame,
        input_columns=["X1", "X2"],
        target_column="Y",
        method="linear",
        random_state=29,
    )
    imputed, reports = impute_missing_values(
        frame,
        input_columns=["X1", "X2"],
        target_columns="Y",
        method="linear",
        random_state=31,
    )
    assert validation.rmse >= 0
    assert not imputed["Y"].isna().any()
    assert "Y" in reports
