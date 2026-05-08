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

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .distance import compute_distance_matrix

COLORS = {
    "candidate": "rgba(51, 51, 51, 0.50)",
    "previous": "rgba(255, 0, 255, 1.0)",
    "design": "rgba(0, 255, 255, 1.0)",
    "imputed": "rgba(255, 0, 0, 0.60)",
}


def suggest_histogram_bins(
    values: pd.Series | np.ndarray,
    *,
    reference: pd.Series | np.ndarray | None = None,
    resolution: float | None = None,
    target_bins: int = 20,
) -> dict[str, float] | None:
    """Suggest readable histogram bin edges for one numeric variable.

    The rule is intentionally simple: use the reference range, divide by a
    target number of bins, and when a resolution is supplied or can be inferred
    snap the width to that step size. The result can be passed directly into
    Plotly's ``xbins`` argument.

    Args:
        values: Values that will be shown in the histogram.
        reference: Optional broader reference values used to infer range or
            resolution. This is useful when plotting a selected design against a
            larger candidate set.
        resolution: Optional meaningful step size for the variable.
        target_bins: Preferred number of bins across the plotted range.

    Returns:
        ``{"start": ..., "end": ..., "size": ...}`` for Plotly histograms, or
        ``None`` when a stable suggestion cannot be formed.
    """

    def _series(data: pd.Series | np.ndarray | None) -> pd.Series | None:
        if data is None:
            return None
        if isinstance(data, pd.Series):
            series = data.copy()
        else:
            array = np.asarray(data).reshape(-1)
            series = pd.Series(array)
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return None
        return finite

    def _infer_resolution(series: pd.Series | None) -> float | None:
        if series is None:
            return None
        unique = np.sort(series.astype(float).unique())
        if len(unique) < 2:
            return None
        diffs = np.diff(unique)
        positive = diffs[diffs > 0]
        if positive.size == 0:
            return None
        return float(np.min(positive))

    values_series = _series(values)
    if values_series is None:
        return None

    base_series = _series(reference)
    if base_series is None:
        base_series = values_series

    lower = float(base_series.min())
    upper = float(base_series.max())
    data_range = upper - lower
    if not math.isfinite(data_range) or data_range <= 0.0:
        return None

    target = max(1, int(target_bins))
    raw_width = data_range / target

    step = resolution
    if step is None:
        step = _infer_resolution(base_series)
    if step is not None:
        step = abs(float(step))
        if step == 0.0 or not math.isfinite(step):
            step = None

    if step is None:
        width = raw_width
    else:
        multiple = max(1, int(round(raw_width / step)))
        width = multiple * step

    if not math.isfinite(width) or width <= 0.0:
        return None

    start = lower - (0.5 * width)
    end = upper + (0.5 * width)
    return {
        "start": float(start),
        "end": float(end),
        "size": float(width),
    }


def suggest_histogram_bin_map(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    reference: pd.DataFrame | None = None,
    resolutions: dict[str, float] | None = None,
    target_bins: int = 20,
) -> dict[str, dict[str, float]]:
    """Suggest histogram bins for several DataFrame columns.

    Args:
        data: Table containing the plotted values.
        columns: Optional subset of columns to process.
        reference: Optional table used to infer broader ranges or resolutions.
        resolutions: Optional explicit per-column resolutions.
        target_bins: Preferred number of bins across each plotted range.

    Returns:
        Mapping from column name to Plotly ``xbins`` dictionaries.
    """

    show = columns or list(data.columns)
    bin_map: dict[str, dict[str, float]] = {}
    for column in show:
        reference_values = None
        if reference is not None and column in reference:
            reference_values = reference[column]
        xbins = suggest_histogram_bins(
            data[column],
            reference=reference_values,
            resolution=None if resolutions is None else resolutions.get(column),
            target_bins=target_bins,
        )
        if xbins is not None:
            bin_map[column] = xbins
    return bin_map


def _finite_series(values: pd.Series | np.ndarray | None) -> pd.Series | None:
    """Return finite numeric values as a Series."""
    if values is None:
        return None
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(np.asarray(values).reshape(-1))
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return None
    return finite


def _axis_range(
    values: list[pd.Series | np.ndarray | None],
    *,
    xbins: dict[str, float] | None = None,
    padding_ratio: float = 0.03,
) -> tuple[float, float] | None:
    """Build a padded numeric axis range."""
    series_parts = [series for series in (_finite_series(value) for value in values) if series is not None]
    if not series_parts:
        return None

    combined = pd.concat(series_parts, ignore_index=True)
    lower = float(combined.min())
    upper = float(combined.max())
    span = upper - lower

    if xbins is not None:
        start = float(xbins.get("start", lower))
        end = float(xbins.get("end", upper))
        size = abs(float(xbins.get("size", 0.0)))
        if math.isfinite(start) and math.isfinite(end) and math.isfinite(size) and end > start and size > 0.0:
            pad = max(0.15 * size, 0.01 * (end - start))
            return (start - pad, end + pad)

    if not math.isfinite(span) or span <= 0.0:
        center = lower
        pad = max(abs(center) * 0.05, 0.5)
        return (center - pad, center + pad)

    pad = max(span * padding_ratio, 1e-9)
    return (lower - pad, upper + pad)


def plot_pair_matrix(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    candidate: pd.DataFrame | None = None,
    previous: pd.DataFrame | None = None,
    title: str = "Design visualization",
    imputed_points: int = 0,
    histogram_bins: dict[str, dict[str, float]] | None = None,
) -> go.Figure:
    """Create a pairwise design-visualization figure.

    Args:
        data: Main design or candidate table to visualize.
        columns: Optional subset of columns to plot.
        candidate: Optional candidate table drawn as a background layer.
        previous: Optional previous-data table drawn as an overlay.
        title: Figure title.
        imputed_points: Number of rows at the end of ``data`` that should be
            highlighted as imputed points.
        histogram_bins: Optional per-column Plotly ``xbins`` settings for the
            diagonal histograms. When omitted, readable bins are inferred.

    Returns:
        Plotly figure with pairwise scatterplots and diagonal histograms.
    """
    legend_state = {
        "Design points": False,
        "Imputed points": False,
        "Candidate points": False,
        "Previous data": False,
    }

    def next_legend(name: str) -> bool:
        """Show each legend entry only on its first plotted trace."""
        show = not legend_state[name]
        legend_state[name] = True
        return show

    show = columns or list(data.columns)
    resolved_histogram_bins = histogram_bins
    if resolved_histogram_bins is None:
        reference = None
        if candidate is not None:
            reference = candidate.loc[:, [column for column in show if column in candidate.columns]]
        elif previous is not None:
            reference = previous.loc[:, [column for column in show if column in previous.columns]]
        resolved_histogram_bins = suggest_histogram_bin_map(
            data.loc[:, show],
            columns=show,
            reference=reference,
        )

    axis_ranges = {
        column: _axis_range(
            [
                data[column],
                None if candidate is None or column not in candidate.columns else candidate[column],
                None if previous is None or column not in previous.columns else previous[column],
            ],
            xbins=None if resolved_histogram_bins is None else resolved_histogram_bins.get(column),
        )
        for column in show
    }
    n = len(show)
    if n == 1:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=data[show[0]],
                name="Design points",
                marker_color=COLORS["design"],
                marker_line={"color": "rgba(0, 0, 0, 0.65)", "width": 0.6},
                opacity=0.75,
                showlegend=next_legend("Design points"),
                xbins=None if resolved_histogram_bins is None else resolved_histogram_bins.get(show[0]),
            )
        )
        if previous is not None:
            fig.add_trace(
                go.Histogram(
                    x=previous[show[0]],
                    name="Previous data",
                    marker_color=COLORS["previous"],
                    marker_line={"color": "rgba(0, 0, 0, 0.65)", "width": 0.6},
                    opacity=0.50,
                    showlegend=next_legend("Previous data"),
                    xbins=None if resolved_histogram_bins is None else resolved_histogram_bins.get(show[0]),
                )
            )
        fig.update_layout(
            title=title,
            barmode="overlay",
            template="plotly_white",
            width=650,
            height=450,
            margin={"l": 110, "r": 40, "t": 105, "b": 70},
        )
        fig.update_xaxes(
            title_text=show[0],
            range=axis_ranges.get(show[0]),
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1.0,
            mirror=True,
            ticks="outside",
            automargin=True,
            title_font={"size": 16},
            title_standoff=6,
        )
        fig.update_yaxes(
            title_text="Frequency",
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1.0,
            mirror=True,
            ticks="outside",
            automargin=True,
            title_font={"size": 16},
            title_standoff=6,
        )
        return fig

    spacing = min(0.09, 0.90 / max(1, n - 1))
    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=spacing,
        vertical_spacing=spacing,
    )
    for row in range(1, n + 1):
        for col in range(1, n + 1):
            x = show[col - 1]
            y = show[row - 1]
            if row == col:
                fig.add_trace(
                    go.Histogram(
                        x=data[x],
                        name="Design points",
                        marker_color=COLORS["design"],
                        marker_line={"color": "rgba(0, 0, 0, 0.65)", "width": 0.6},
                        opacity=0.75,
                        showlegend=next_legend("Design points"),
                        xbins=None if resolved_histogram_bins is None else resolved_histogram_bins.get(x),
                    ),
                    row=row,
                    col=col,
                )
                if previous is not None:
                    fig.add_trace(
                        go.Histogram(
                            x=previous[x],
                            name="Previous data",
                            marker_color=COLORS["previous"],
                            marker_line={"color": "rgba(0, 0, 0, 0.65)", "width": 0.6},
                            opacity=0.50,
                            showlegend=next_legend("Previous data"),
                            xbins=None if resolved_histogram_bins is None else resolved_histogram_bins.get(x),
                        ),
                        row=row,
                        col=col,
                    )
            elif row > col:
                if candidate is not None:
                    fig.add_trace(
                        go.Scattergl(
                            x=candidate[x],
                            y=candidate[y],
                            mode="markers",
                            name="Candidate points",
                            marker={"color": COLORS["candidate"], "size": 5},
                            showlegend=next_legend("Candidate points"),
                        ),
                        row=row,
                        col=col,
                    )
                if previous is not None:
                    fig.add_trace(
                        go.Scattergl(
                            x=previous[x],
                            y=previous[y],
                            mode="markers",
                            name="Previous data",
                            marker={"color": COLORS["previous"], "size": 5},
                            showlegend=next_legend("Previous data"),
                        ),
                        row=row,
                        col=col,
                    )
                if imputed_points > 0:
                    observed = data.iloc[:-imputed_points]
                    imputed = data.iloc[-imputed_points:]
                    fig.add_trace(
                        go.Scattergl(
                            x=observed[x],
                            y=observed[y],
                            mode="markers",
                            name="Design points",
                            marker={"color": COLORS["design"], "size": 6},
                            showlegend=next_legend("Design points"),
                        ),
                        row=row,
                        col=col,
                    )
                    fig.add_trace(
                        go.Scattergl(
                            x=imputed[x],
                            y=imputed[y],
                            mode="markers",
                            name="Imputed points",
                            marker={"color": COLORS["imputed"], "size": 6},
                            showlegend=next_legend("Imputed points"),
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    fig.add_trace(
                        go.Scattergl(
                            x=data[x],
                            y=data[y],
                            mode="markers",
                            name="Design points",
                            marker={"color": COLORS["design"], "size": 6},
                            showlegend=next_legend("Design points"),
                        ),
                        row=row,
                        col=col,
                    )
            else:
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue

            fig.update_xaxes(
                title_text=x,
                range=axis_ranges.get(x),
                showgrid=True,
                zeroline=False,
                automargin=True,
                showline=True,
                linecolor="black",
                linewidth=1.0,
                mirror=True,
                ticks="outside",
                title_font={"size": 16},
                title_standoff=6,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Frequency" if row == col else y,
                range=None if row == col else axis_ranges.get(y),
                showgrid=True,
                zeroline=False,
                automargin=True,
                showline=True,
                linecolor="black",
                linewidth=1.0,
                mirror=True,
                ticks="outside",
                title_font={"size": 16},
                title_standoff=6,
                row=row,
                col=col,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        barmode="overlay",
        width=max(760, 320 * n),
        height=max(760, 320 * n),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 90, "r": 40, "t": 95, "b": 70},
    )
    return fig


def plot_nonuniform_weights(
    scaled_design_inputs: np.ndarray,
    scaled_design_weights: np.ndarray,
    candidate_weights: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    """Plot weighted-distance diagnostics for a non-uniform design.

    Args:
        scaled_design_inputs: Scaled input coordinates of the chosen design.
        scaled_design_weights: Scaled weights attached to design rows.
        candidate_weights: Scaled weights for the full candidate set.
        title: Figure title.

    Returns:
        Plotly figure showing nearest-neighbor distances and the candidate
        weight distribution.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"Closest-neighbor distance (N={len(scaled_design_weights)})",
            "Candidate weight distribution",
        ),
    )
    distance_matrix = compute_distance_matrix(scaled_design_inputs)
    minimum_distance = np.sqrt(np.min(distance_matrix, axis=1))
    for weight, distance in zip(scaled_design_weights, minimum_distance):
        fig.add_trace(
            go.Scatter(
                x=[weight, weight],
                y=[0, distance],
                mode="lines",
                line={"color": "rgba(0, 90, 255, 0.80)", "width": 2},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=candidate_weights,
            marker_color=COLORS["design"],
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=700,
    )
    fig.update_yaxes(title_text="Min distance", row=1, col=1)
    fig.update_xaxes(title_text="Scaled weight", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    return fig


def plot_pareto_front(pareto_front: pd.DataFrame) -> go.Figure:
    """Plot the IRSF Pareto front.

    Args:
        pareto_front: Table with ``Best Input`` and ``Best Response`` columns.

    Returns:
        Plotly figure of the Pareto front in input-response criterion space.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pareto_front["Best Input"],
            y=pareto_front["Best Response"],
            mode="lines+markers",
            line={"color": "black", "width": 1},
            marker={"color": "rgba(31, 119, 180, 1.0)", "symbol": "diamond", "size": 9},
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Pareto front",
        template="plotly_white",
        width=850,
        height=550,
    )
    fig.update_xaxes(title_text="Maximin input distance", showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Maximin response distance", showgrid=True, zeroline=False)
    return fig


def write_figure(figure: go.Figure, path: str | Path) -> None:
    """Export a Plotly figure to a static file.

    Args:
        figure: Figure to export.
        path: Output path. The suffix controls the export format.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = output_path.suffix.lstrip(".").lower() or "pdf"
    figure.write_image(str(output_path), format=image_format)
