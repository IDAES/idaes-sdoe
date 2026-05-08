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
from idaes_sdoe.design.uniform import design_uniform, design_uniform_batch, estimate_uniform_runtime


FIXTURES = Path(__file__).parent / "fixtures"


def _setup():
    candidate = load_csv(FIXTURES / "candidate.csv")
    previous = load_csv(FIXTURES / "previous.csv")
    return prepare_design_setup(
        candidate=candidate,
        previous=previous,
        roles=ColumnRoles(inputs=["X1", "X2"], weight="Weight", responses=["Y"]),
    )


def test_uniform_design_smoke():
    result = design_uniform(
        setup=_setup(),
        design_size=3,
        num_restarts=20,
        mode="maximin",
        random_state=7,
    )
    assert len(result.design) == 3
    assert result.mode == "maximin"


def test_uniform_batch_and_estimate():
    setup = _setup()
    batch = design_uniform_batch(
        setup=setup,
        design_sizes=[2, 3],
        num_restarts=10,
        mode="minimax",
        random_state=11,
    )
    estimate = estimate_uniform_runtime(
        setup=setup,
        design_sizes=[2, 3],
        target_restarts=50,
        random_state=3,
    )
    assert len(batch) == 2
    assert estimate["total_seconds"] > 0
