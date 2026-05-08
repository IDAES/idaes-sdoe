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
from idaes_sdoe.io import aggregate_tables


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_and_aggregate():
    candidate = load_csv(FIXTURES / "candidate.csv")
    aggregated = aggregate_tables([FIXTURES / "candidate.csv", FIXTURES / "candidate.csv"])
    assert list(candidate.columns) == ["X1", "X2", "Weight", "Y"]
    assert len(aggregated) == len(candidate)


def test_prepare_design_setup_adds_index():
    candidate = load_csv(FIXTURES / "candidate.csv")
    previous = load_csv(FIXTURES / "previous.csv")
    setup = prepare_design_setup(
        candidate=candidate,
        previous=previous,
        roles=ColumnRoles(inputs=["X1", "X2"], responses=["Y"], weight="Weight"),
    )
    assert setup.roles.index == "__id"
    assert "__id" in setup.candidate.columns
    assert "__id" in setup.previous.columns
