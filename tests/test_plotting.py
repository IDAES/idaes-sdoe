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
import pytest

from idaes_sdoe.plotting import plot_pair_matrix, suggest_histogram_bin_map, suggest_histogram_bins


def test_suggest_histogram_bins_snaps_to_resolution():
    values = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0])
    reference = pd.Series([round(-1.0 + (0.05 * i), 2) for i in range(41)])

    bins = suggest_histogram_bins(
        values,
        reference=reference,
        target_bins=20,
    )

    assert bins is not None
    assert bins["size"] == pytest.approx(0.1)
    assert bins["start"] == pytest.approx(-1.05)
    assert bins["end"] == pytest.approx(1.05)


def test_plot_pair_matrix_uses_supplied_histogram_bins():
    frame = pd.DataFrame(
        {
            "X1": [-1.0, 0.0, 1.0],
            "X2": [-1.0, 0.0, 1.0],
        }
    )
    bin_map = suggest_histogram_bin_map(frame, target_bins=10)
    figure = plot_pair_matrix(frame, columns=["X1", "X2"], histogram_bins=bin_map)

    histogram_traces = [trace for trace in figure.data if trace.type == "histogram"]
    assert len(histogram_traces) == 2
    assert histogram_traces[0].xbins.size is not None
    assert histogram_traces[1].xbins.size is not None


def test_plot_pair_matrix_labels_visible_axes():
    frame = pd.DataFrame(
        {
            "X1": [-1.0, 0.0, 1.0],
            "X2": [-1.0, 0.0, 1.0],
        }
    )

    figure = plot_pair_matrix(frame, columns=["X1", "X2"])

    assert figure.layout.xaxis.title.text == "X1"
    assert figure.layout.yaxis.title.text == "Frequency"
    assert figure.layout.xaxis.range[0] < -1.0
    assert figure.layout.xaxis.range[1] > 1.0
    assert figure.layout.xaxis3.title.text == "X1"
    assert figure.layout.yaxis3.title.text == "X2"
    assert figure.layout.xaxis3.range[0] < -1.0
    assert figure.layout.xaxis3.range[1] > 1.0
    assert figure.layout.xaxis4.title.text == "X2"
    assert figure.layout.yaxis4.title.text == "Frequency"
    assert len(figure.layout.annotations) == 0
