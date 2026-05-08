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
from pathlib import Path

from idaes_sdoe import ColumnRoles, load_csv, prepare_design_setup
from idaes_sdoe.design.input_response import design_input_response, estimate_input_response_runtime


FIXTURES = Path(__file__).parent / "fixtures"


def _setup():
    candidate = load_csv(FIXTURES / "candidate.csv")
    previous = load_csv(FIXTURES / "previous.csv")
    return prepare_design_setup(
        candidate=candidate,
        previous=previous,
        roles=ColumnRoles(inputs=["X1", "X2"], responses=["Y"], weight="Weight"),
    )


def test_input_response_design_smoke():
    result = design_input_response(
        setup=_setup(),
        design_size=3,
        num_restarts=3,
        random_state=17,
        weight_grid=[0.3, 0.7],
    )
    assert result.num_designs >= 1
    assert not result.pareto_front.empty


def test_input_response_estimate():
    estimate = estimate_input_response_runtime(
        setup=_setup(),
        design_size=3,
        target_restarts=6,
        calibration_restarts=2,
        random_state=19,
        weight_grid=[0.5],
    )
    assert estimate["estimated_seconds"] > 0
