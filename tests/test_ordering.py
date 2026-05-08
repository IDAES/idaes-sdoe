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

from idaes_sdoe import load_csv
from idaes_sdoe.ordering import order_runs


FIXTURES = Path(__file__).parent / "fixtures"


def test_order_runs_smoke():
    design = load_csv(FIXTURES / "candidate.csv").iloc[:4].copy()
    result = order_runs(design, input_columns=["X1", "X2"], exact=False)
    assert len(result.permutation) == 4
    assert len(result.ordered_design) == 4
    assert result.path_length >= 0
