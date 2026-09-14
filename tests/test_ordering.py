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

import pandas as pd

from idaes_sdoe import load_csv
from idaes_sdoe.ordering import order_runs, order_runs_by_difficulty


FIXTURES = Path(__file__).parent / "fixtures"


def test_order_runs_smoke():
    design = load_csv(FIXTURES / "candidate.csv").iloc[:4].copy()
    result = order_runs(design, input_columns=["X1", "X2"], exact=False)
    assert len(result.permutation) == 4
    assert len(result.ordered_design) == 4
    assert result.path_length >= 0


def test_order_by_difficulty_blocks_sorts_hard_columns():
    design = pd.DataFrame({"X1": [3.0, 1.0, 2.0, 1.0], "X2": [0.1, 0.2, 0.3, 0.05]})
    result = order_runs_by_difficulty(
        design, input_columns=["X1", "X2"], hard_columns=["X1"]
    )
    assert result.method == "difficulty-blocks"
    assert list(result.ordered_design["X1"]) == sorted(design["X1"])
    assert len(result.permutation) == 4


def test_order_by_difficulty_all_hard_uses_tsp():
    design = pd.DataFrame({"X1": [0.0, 1.0, 0.5], "X2": [0.0, 1.0, 0.5]})
    result = order_runs_by_difficulty(
        design, input_columns=["X1", "X2"], hard_columns=["X1", "X2"], exact=False
    )
    assert result.method.startswith("tsp")


def test_order_by_difficulty_none_hard_keeps_order():
    design = pd.DataFrame({"X1": [3.0, 1.0, 2.0], "X2": [1.0, 2.0, 3.0]})
    result = order_runs_by_difficulty(
        design, input_columns=["X1", "X2"], hard_columns=[]
    )
    assert result.permutation == [0, 1, 2]
    assert result.method == "original"
