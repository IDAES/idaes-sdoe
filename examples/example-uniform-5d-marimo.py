import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import json
    import re
    import sys
    import time
    import zipfile
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    def _find_project_root() -> Path:
        starts = [Path.cwd().resolve()]
        _notebook_dir = mo.notebook_dir()
        if _notebook_dir is not None:
            starts.insert(0, Path(_notebook_dir).resolve())
        for _start in starts:
            for _path in (_start, *_start.parents):
                if (_path / "src" / "idaes_sdoe").is_dir():
                    return _path
        raise RuntimeError(
            "Could not locate the idaes-sdoe project root. "
            "Start marimo from this repository or keep this notebook under examples/."
        )

    ROOT = _find_project_root()
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from idaes_sdoe import ColumnRoles, prepare_design_setup, write_csv
    from idaes_sdoe.design import design_uniform_batch, estimate_uniform_runtime
    from idaes_sdoe.plotting import (
        plot_pair_matrix,
        suggest_histogram_bin_map,
        write_figure,
    )

    DEFAULT_INPUT_COLUMNS = ["G", "lldg", "CapturePerc", "L", "SteamFlow"]
    DEFAULT_HISTOGRAM_BINS = 25
    MAX_SCATTER_PLOTS = 12
    MAX_HISTOGRAMS = 12
    RESTART_CHOICES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    MAX_RANDOM_SEED = 4_294_967_295
    FIGURE_FORMATS = ["pdf", "png", "svg", "html"]
    FIGURE_MIME_TYPES = {
        "html": "text/html",
        "pdf": "application/pdf",
        "png": "image/png",
        "svg": "image/svg+xml",
    }
    CANDIDATE_POINT_COLOR = "rgba(51, 51, 51, 0.50)"
    DESIGN_POINT_COLOR = "rgba(0, 255, 255, 1.0)"
    PREVIOUS_POINT_COLOR = "rgba(255, 0, 255, 1.0)"
    AUTO_INDEX = "__AUTO_INDEX__"
    SESSION_STAMP = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    DEFAULT_OUTPUT_DIR = (
        ROOT
        / "examples"
        / "temp"
        / "marimo_output"
        / f"{SESSION_STAMP}_example_uniform_5d"
    )
    return (
        AUTO_INDEX,
        CANDIDATE_POINT_COLOR,
        ColumnRoles,
        DEFAULT_HISTOGRAM_BINS,
        DEFAULT_INPUT_COLUMNS,
        DEFAULT_OUTPUT_DIR,
        DESIGN_POINT_COLOR,
        FIGURE_FORMATS,
        FIGURE_MIME_TYPES,
        MAX_HISTOGRAMS,
        MAX_RANDOM_SEED,
        MAX_SCATTER_PLOTS,
        PREVIOUS_POINT_COLOR,
        Path,
        RESTART_CHOICES,
        ROOT,
        datetime,
        design_uniform_batch,
        estimate_uniform_runtime,
        go,
        io,
        json,
        mo,
        np,
        pd,
        plot_pair_matrix,
        prepare_design_setup,
        re,
        suggest_histogram_bin_map,
        time,
        write_csv,
        write_figure,
        zipfile,
    )


@app.cell
def _(
    DEFAULT_HISTOGRAM_BINS,
    Path,
    ROOT,
    datetime,
    estimate_uniform_runtime,
    io,
    json,
    np,
    pd,
    plot_pair_matrix,
    re,
    suggest_histogram_bin_map,
    write_figure,
    zipfile,
):
    def resolve_output_directory(_value: str):
        """Resolve and validate a user-supplied artifact directory."""
        _cleaned = str(_value).strip()
        if not _cleaned:
            raise ValueError("Enter an output directory.")
        _requested = Path(_cleaned).expanduser()
        _resolved = (
            _requested
            if _requested.is_absolute()
            else ROOT / _requested
        ).resolve()
        _filesystem_root = Path(_resolved.anchor).resolve()
        if _resolved in {_filesystem_root, ROOT.resolve()}:
            raise ValueError(
                "Choose a dedicated output subdirectory, not the filesystem "
                "or repository root."
            )
        if _resolved.exists() and not _resolved.is_dir():
            raise ValueError(
                f"The output path exists and is not a directory: {_resolved}"
            )
        return _resolved

    def normalize_columns(frame):
        """Strip header whitespace and apply the example's SteamFlow alias."""
        _frame = frame.copy()
        _frame.rename(columns=lambda _name: str(_name).strip(), inplace=True)
        if "Steam Flow" in _frame.columns and "SteamFlow" not in _frame.columns:
            _frame.rename(columns={"Steam Flow": "SteamFlow"}, inplace=True)
        _duplicates = _frame.columns[_frame.columns.duplicated()].tolist()
        if _duplicates:
            raise ValueError(
                "Column normalization produced duplicate names: "
                + ", ".join(map(str, _duplicates))
            )
        return _frame

    def read_uploaded_csv(_upload):
        return normalize_columns(pd.read_csv(io.BytesIO(_upload.contents)))

    def editor_frame(_value):
        if isinstance(_value, pd.DataFrame):
            return _value.copy()
        return pd.DataFrame(_value)

    def clean_manual_candidate(_value):
        """Convert the editable grid into a compact, typed candidate table."""
        _frame = normalize_columns(editor_frame(_value))
        _frame = (
            _frame.replace(r"^\s*$", pd.NA, regex=True)
            .dropna(how="all")
            .reset_index(drop=True)
        )
        for _name in _frame.columns:
            _nonmissing = _frame[_name].notna()
            if not _nonmissing.any():
                continue
            _numeric = pd.to_numeric(_frame[_name], errors="coerce")
            if _numeric.loc[_nonmissing].notna().all():
                _frame[_name] = _numeric
        return _frame

    def bounds_from_editor(
        _value, _input_columns: list[str]
    ) -> dict[str, tuple[float, float]]:
        _table = editor_frame(_value)
        _required = {"input", "minimum", "maximum"}
        if not _required.issubset(_table.columns):
            raise ValueError("The bounds editor must contain input/minimum/maximum.")
        if _table["input"].tolist() != _input_columns:
            raise ValueError(
                "The bounds rows changed unexpectedly. Re-select the input columns "
                "to reset the editor."
            )
        _bounds: dict[str, tuple[float, float]] = {}
        for _row in _table.itertuples(index=False):
            _lower = float(_row.minimum)
            _upper = float(_row.maximum)
            if not np.isfinite(_lower) or not np.isfinite(_upper):
                raise ValueError(f"Bounds for {_row.input!r} must be finite.")
            if _lower >= _upper:
                raise ValueError(
                    f"Minimum must be below maximum for {_row.input!r}."
                )
            _bounds[str(_row.input)] = (_lower, _upper)
        return _bounds

    def format_duration(_seconds: float) -> str:
        _seconds = max(0.0, float(_seconds))
        if _seconds < 1:
            return f"{_seconds * 1000:.0f} ms"
        if _seconds < 60:
            return f"{_seconds:.1f} s"
        _minutes, _remainder = divmod(round(_seconds), 60)
        if _minutes < 60:
            return f"{_minutes} min {_remainder} s"
        _hours, _minutes = divmod(_minutes, 60)
        return f"{_hours} hr {_minutes} min"

    def write_json(_payload: dict[str, object], _path) -> None:
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open("w", encoding="utf-8") as _handle:
            json.dump(_payload, _handle, indent=2)

    def save_figure(
        _figure, _path
    ):
        try:
            write_figure(_figure, _path)
        except Exception as _exc:
            _error_summary = (
                str(_exc).splitlines()[0] or type(_exc).__name__
            )
            _fallback = _path.with_suffix(".html")
            try:
                _figure.write_html(
                    _fallback,
                    include_plotlyjs="cdn",
                    full_html=True,
                )
            except Exception as _fallback_exc:
                return (
                    None,
                    (
                        f"Could not export {_path.name}: {_error_summary}. "
                        f"HTML fallback also failed: {_fallback_exc}"
                    ),
                )
            return (
                _fallback,
                (
                    f"Could not export {_path.name}; saved interactive fallback "
                    f"{_fallback.name}. Static exporter error: {_error_summary}"
                ),
            )
        return _path, None

    def csv_bytes(_frame) -> bytes:
        return _frame.to_csv(index=False).encode("utf-8")

    def safe_filename_component(_value: object) -> str:
        """Convert a plot label into a readable, portable filename component."""
        _cleaned = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            str(_value).strip(),
        ).strip("._-")
        return _cleaned or "variable"

    def zip_directory(_directory) -> bytes:
        _buffer = io.BytesIO()
        with zipfile.ZipFile(
            _buffer, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as _archive:
            if _directory.exists():
                for _path in sorted(_directory.rglob("*")):
                    if _path.is_file():
                        _archive.write(_path, _path.relative_to(_directory))
        return _buffer.getvalue()

    def new_run_stamp() -> str:
        return datetime.now().strftime("%Y%m%dT%H%M%S_%f")

    def validate_candidate(
        _frame,
        _inputs: list[str],
        _index: str | None,
    ) -> None:
        if len(_frame) < 2:
            raise ValueError("The candidate table must contain at least two rows.")
        if not _inputs:
            raise ValueError("Select at least one input column.")
        _missing = [_name for _name in _inputs if _name not in _frame.columns]
        if _missing:
            raise ValueError("Missing input columns: " + ", ".join(_missing))
        _not_numeric = [
            _name
            for _name in _inputs
            if not pd.api.types.is_numeric_dtype(_frame[_name])
        ]
        if _not_numeric:
            raise ValueError(
                "Design inputs must be numeric: " + ", ".join(_not_numeric)
            )
        _numeric = _frame[_inputs].to_numpy(dtype=float)
        if not np.isfinite(_numeric).all():
            raise ValueError("Selected inputs contain missing or non-finite values.")
        _zero_width = [
            _name
            for _name in _inputs
            if float(_frame[_name].min()) == float(_frame[_name].max())
        ]
        if _zero_width:
            raise ValueError(
                "Inputs must vary across candidates: " + ", ".join(_zero_width)
            )
        if _index is not None:
            if _index in _inputs:
                raise ValueError("The index column cannot also be a design input.")
            if _frame[_index].isna().any():
                raise ValueError(f"Index column {_index!r} contains missing values.")
            if not _frame[_index].is_unique:
                raise ValueError(f"Index column {_index!r} must be unique.")
            if not pd.api.types.is_numeric_dtype(_frame[_index]):
                raise ValueError(
                    "The current API requires a numeric index column. "
                    "Choose a numeric column or generate __id."
                )

    def align_previous_frame(
        _previous,
        _candidate,
        _roles,
    ):
        """Align uploaded history while preserving all candidate output columns."""
        _previous = normalize_columns(_previous)
        _notes: list[str] = []
        _required_inputs = list(_roles.inputs)
        _missing_inputs = [
            _name for _name in _required_inputs if _name not in _previous.columns
        ]
        if _missing_inputs:
            raise ValueError(
                "Previous data is missing design inputs: "
                + ", ".join(_missing_inputs)
            )

        _index = _roles.index
        _replace_index = False
        _index_collision = False
        if _index and _index in _previous.columns:
            _numeric_index = pd.to_numeric(
                _previous[_index], errors="coerce"
            )
            _replace_index = (
                _numeric_index.isna().any()
                or not _numeric_index.is_unique
            )
            if not _replace_index:
                _candidate_ids = set(
                    pd.to_numeric(
                        _candidate[_index], errors="coerce"
                    ).dropna()
                )
                _index_collision = bool(
                    set(_numeric_index.tolist()) & _candidate_ids
                )
                _replace_index = _index_collision
                if not _replace_index:
                    _previous[_index] = _numeric_index
        if _index and (
            _index not in _previous.columns or _replace_index
        ):
            _candidate_index = pd.to_numeric(
                _candidate[_index], errors="coerce"
            ).dropna()
            _start = (
                int(_candidate_index.max()) + 1
                if not _candidate_index.empty
                else len(_candidate)
            )
            if _index in _previous.columns:
                _previous[_index] = range(
                    _start, _start + len(_previous)
                )
                _notes.append(
                    (
                        f"Remapped uploaded {_index} values into a disjoint "
                        "range because they overlap candidate identifiers."
                        if _index_collision
                        else (
                            f"Replaced non-numeric or duplicate {_index} "
                            "values in uploaded previous data."
                        )
                    )
                )
            else:
                _previous.insert(
                    0,
                    _index,
                    range(_start, _start + len(_previous)),
                )
                _notes.append(
                    f"Generated {_index} for uploaded previous data."
                )

        _missing_excluded = [
            _name
            for _name in _candidate.columns
            if _name not in _previous.columns
        ]
        for _name in _missing_excluded:
            _previous[_name] = pd.NA
        if _missing_excluded:
            _notes.append(
                "Filled missing excluded columns with blank values: "
                + ", ".join(_missing_excluded)
            )

        _extra = [
            _name
            for _name in _previous.columns
            if _name not in _candidate.columns
        ]
        if _extra:
            _notes.append("Ignored extra columns: " + ", ".join(_extra))

        _aligned = _previous.loc[:, _candidate.columns].copy()
        _aligned = _aligned.drop_duplicates().reset_index(drop=True)
        _numeric = _aligned[_required_inputs].apply(
            pd.to_numeric, errors="coerce"
        )
        if _numeric.isna().any().any():
            raise ValueError(
                "Previous-data inputs must be numeric and non-missing."
            )
        _aligned[_required_inputs] = _numeric
        return _aligned, _notes

    def exclude_previous_candidates(
        _candidate,
        _previous,
        _roles,
        *,
        trust_index: bool,
    ):
        _input_keys = set(
            _previous[_roles.inputs].itertuples(index=False, name=None)
        )
        _input_overlap = pd.Series(
            [
                _row in _input_keys
                for _row in _candidate[_roles.inputs].itertuples(
                    index=False, name=None
                )
            ],
            index=_candidate.index,
        )
        _index_overlap = pd.Series(False, index=_candidate.index)
        if trust_index and _roles.index:
            _previous_ids = set(_previous[_roles.index].tolist())
            _index_overlap = _candidate[_roles.index].isin(_previous_ids)
        _remove = _input_overlap | _index_overlap
        _remaining = _candidate.loc[~_remove].copy().reset_index(drop=True)
        return _remaining, {
            "removed": int(_remove.sum()),
            "matched_by_inputs": int(_input_overlap.sum()),
            "matched_by_index": int(_index_overlap.sum()),
        }

    def pair_figure(
        _data,
        _columns: list[str],
        *,
        title: str,
        candidate=None,
        previous=None,
        target_bins: int = DEFAULT_HISTOGRAM_BINS,
    ):
        _reference = candidate if candidate is not None else previous
        _bins = suggest_histogram_bin_map(
            _data[_columns],
            columns=_columns,
            reference=(
                None
                if _reference is None
                else _reference.loc[:, _columns]
            ),
            target_bins=target_bins,
        )
        return plot_pair_matrix(
            _data[_columns],
            columns=_columns,
            candidate=(
                None
                if candidate is None
                else candidate.loc[:, _columns]
            ),
            previous=(
                None
                if previous is None
                else previous.loc[:, _columns]
            ),
            title=title,
            histogram_bins=_bins,
        )

    def create_run_dirs(
        _output_dir,
        _prefix: str,
        _mode: str,
        _stamp: str,
    ):
        _run_dir = _output_dir / f"{_prefix}_{_mode}_{_stamp}"
        _design_dir = _run_dir / "designs"
        _plot_dir = _run_dir / "plots"
        _design_dir.mkdir(parents=True, exist_ok=True)
        _plot_dir.mkdir(parents=True, exist_ok=True)
        return _run_dir, _design_dir, _plot_dir

    def write_runtime_estimate(
        _setup,
        _design_sizes: list[int],
        _config: dict[str, object],
        _run_dir,
    ) -> dict[str, object]:
        _payload = estimate_uniform_runtime(
            setup=_setup,
            design_sizes=_design_sizes,
            target_restarts=_config["num_restarts"],
            mode=_config["mode"],
            calibration_restarts=_config["calibration_restarts"],
            random_state=_config["random_seed"],
        )
        write_json(_payload, _run_dir / "runtime_estimate.json")
        return _payload

    def save_figure_artifact(
        _figure,
        _path,
        _warnings: list[str],
        *,
        relative_to=None,
    ) -> str:
        _saved_path, _warning = save_figure(_figure, _path)
        if _warning:
            _warnings.append(_warning)
        if _saved_path is None or relative_to is None:
            return ""
        return str(_saved_path.relative_to(relative_to))

    return (
        align_previous_frame,
        bounds_from_editor,
        clean_manual_candidate,
        create_run_dirs,
        csv_bytes,
        exclude_previous_candidates,
        format_duration,
        new_run_stamp,
        pair_figure,
        read_uploaded_csv,
        resolve_output_directory,
        safe_filename_component,
        save_figure_artifact,
        validate_candidate,
        write_json,
        write_runtime_estimate,
        zip_directory,
    )


@app.cell
def _(MAX_RANDOM_SEED, RESTART_CHOICES, mo):
    def make_common_stage_widgets(_seed_value: int):
        return {
            "criterion": mo.ui.radio(
                {"Minimax": "minimax", "Maximin": "maximin"},
                value="Minimax",
                inline=True,
            ),
            "restarts": mo.ui.slider(
                steps=RESTART_CHOICES,
                value=10_000,
                show_value=True,
            ),
            "calibration_restarts": mo.ui.number(
                start=1,
                stop=100_000,
                step=1,
                value=100,
            ),
            "random_seed": mo.ui.number(
                start=0,
                stop=MAX_RANDOM_SEED,
                step=1,
                value=_seed_value,
            ),
        }

    return (make_common_stage_widgets,)


@app.cell
def _(
    FIGURE_FORMATS,
    FIGURE_MIME_TYPES,
    MAX_HISTOGRAMS,
    MAX_SCATTER_PLOTS,
    go,
    io,
    mo,
    safe_filename_component,
    suggest_histogram_bin_map,
    time,
    zipfile,
):
    def make_visualization_action_widgets(_scope: str):
        _format_widget = mo.ui.dropdown(
            options=FIGURE_FORMATS,
            value="svg",
            label="Figure format",
            full_width=True,
        )
        _pair_select_all = mo.ui.button(
            value=0,
            on_click=lambda _value: time.monotonic_ns(),
            label="Select all",
            tooltip=f"Select every {_scope} pairwise scatter plot",
        )
        _pair_clear_all = mo.ui.button(
            value=0,
            on_click=lambda _value: time.monotonic_ns(),
            label="Clear all",
            tooltip=f"Clear all {_scope} pairwise scatter selections",
        )
        _histogram_select_all = mo.ui.button(
            value=0,
            on_click=lambda _value: time.monotonic_ns(),
            label="Select all",
            tooltip=f"Select every {_scope} marginal histogram",
        )
        _histogram_clear_all = mo.ui.button(
            value=0,
            on_click=lambda _value: time.monotonic_ns(),
            label="Clear all",
            tooltip=f"Clear all {_scope} histogram selections",
        )
        return (
            _format_widget,
            _pair_select_all,
            _pair_clear_all,
            _histogram_select_all,
            _histogram_clear_all,
        )

    def make_visualization_count_widgets(
        _input_columns: list[str],
        _scope: str,
    ):
        _available_pair_count = (
            len(_input_columns) * (len(_input_columns) - 1) // 2
        )
        _pair_limit = min(
            _available_pair_count,
            MAX_SCATTER_PLOTS,
        )
        _histogram_limit = min(
            len(_input_columns),
            MAX_HISTOGRAMS,
        )
        _pair_count_widget = mo.ui.number(
            start=0,
            stop=max(1, _pair_limit),
            step=1,
            value=0,
            debounce=True,
            label=f"Number of {_scope} pairwise scatter plots",
            disabled=_pair_limit == 0,
        )
        _histogram_count_widget = mo.ui.number(
            start=0,
            stop=max(1, _histogram_limit),
            step=1,
            value=0,
            debounce=True,
            label=f"Number of {_scope} marginal histograms",
            disabled=_histogram_limit == 0,
        )
        return (
            _available_pair_count,
            _pair_limit,
            _histogram_limit,
            _pair_count_widget,
            _histogram_count_widget,
        )

    def make_visualization_selectors(
        _input_columns: list[str],
        _pair_count: int,
        _histogram_count: int,
    ):
        _available_pairs = [
            (_x_column, _y_column)
            for _x_position, _x_column in enumerate(_input_columns)
            for _y_column in _input_columns[_x_position + 1 :]
        ]
        _shown_pairs = _available_pairs[: max(0, int(_pair_count))]
        _pair_x_selectors = mo.ui.array(
            [
                mo.ui.dropdown(
                    options=_input_columns,
                    value=_x_column,
                    label=f"Scatter plot {_position}: x-axis",
                    full_width=True,
                )
                for _position, (_x_column, _y_column) in enumerate(
                    _shown_pairs,
                    start=1,
                )
            ]
        )
        _pair_y_selectors = mo.ui.array(
            [
                mo.ui.dropdown(
                    options=_input_columns,
                    value=_y_column,
                    label=f"Scatter plot {_position}: y-axis",
                    full_width=True,
                )
                for _position, (_x_column, _y_column) in enumerate(
                    _shown_pairs,
                    start=1,
                )
            ]
        )
        _histogram_selectors = mo.ui.array(
            [
                mo.ui.dropdown(
                    options=_input_columns,
                    value=_input_columns[_position],
                    label=f"Histogram {_position + 1}: variable",
                    full_width=True,
                )
                for _position in range(
                    min(
                        max(0, int(_histogram_count)),
                        len(_input_columns),
                    )
                )
            ]
        )
        return (
            _pair_x_selectors,
            _pair_y_selectors,
            _histogram_selectors,
        )

    def make_visualization_save_selectors(
        _count: int,
        _select_all_button,
        _clear_all_button,
    ):
        _bulk_selected = (
            _select_all_button.value > _clear_all_button.value
        )
        return mo.ui.array(
            [
                mo.ui.checkbox(value=_bulk_selected, label="")
                for _position in range(max(0, int(_count)))
            ]
        )

    def _apply_plot_axes(_figure):
        _axis_style = {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
            "mirror": True,
            "ticks": "outside",
            "automargin": True,
        }
        _figure.update_xaxes(**_axis_style)
        _figure.update_yaxes(**_axis_style)
        return _figure

    def visualization_scatter_figure(
        _series: list[dict[str, object]],
        _x_column: str,
        _y_column: str,
        *,
        hover_include_name: bool = True,
    ):
        _figure = go.Figure()
        for _item in _series:
            _data = _item["data"]
            _name = str(_item["name"])
            _hovertemplate = (
                f"{_name}<br>"
                f"{_x_column}: %{{x}}<br>"
                f"{_y_column}: %{{y}}<extra></extra>"
                if hover_include_name
                else (
                    f"{_x_column}: %{{x}}<br>"
                    f"{_y_column}: %{{y}}<extra></extra>"
                )
            )
            _figure.add_trace(
                go.Scattergl(
                    x=_data[_x_column],
                    y=_data[_y_column],
                    mode="markers",
                    name=_name,
                    legendgroup=_name,
                    marker={
                        "color": _item["color"],
                        "size": int(_item.get("size", 6)),
                    },
                    showlegend=False,
                    hovertemplate=_hovertemplate,
                )
            )
        _figure.update_layout(
            title={
                "text": f"{_y_column} versus {_x_column}",
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title=_x_column,
            yaxis_title=_y_column,
            width=330,
            height=330,
            autosize=False,
            margin={"l": 60, "r": 20, "t": 60, "b": 55},
            template="plotly_white",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return _apply_plot_axes(_figure)

    def visualization_histogram_figure(
        _series: list[dict[str, object]],
        _column: str,
        _target_bins: int,
        *,
        reference,
        hover_include_name: bool = True,
        count_label: str = "Count",
    ):
        _primary_data = _series[0]["data"]
        _bin_map = suggest_histogram_bin_map(
            _primary_data[[_column]],
            columns=[_column],
            reference=reference[[_column]],
            target_bins=_target_bins,
        )
        _figure = go.Figure()
        for _item in _series:
            _name = str(_item["name"])
            _trace_options = {
                "x": _item["data"][_column],
                "name": _name,
                "legendgroup": _name,
                "marker_color": _item["color"],
                "marker_line": {
                    "color": "rgba(0, 0, 0, 0.65)",
                    "width": 0.6,
                },
                "opacity": float(_item.get("opacity", 0.75)),
                "showlegend": False,
                "hovertemplate": (
                    f"{_name}<br>Count: %{{y}}<extra></extra>"
                    if hover_include_name
                    else "Count: %{y}<extra></extra>"
                ),
            }
            if _column in _bin_map:
                _trace_options["xbins"] = _bin_map[_column]
            else:
                _trace_options["nbinsx"] = _target_bins
            _figure.add_trace(go.Histogram(**_trace_options))
        _figure.update_layout(
            title={
                "text": f"Marginal distribution of {_column}",
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title=_column,
            yaxis_title=count_label,
            height=390,
            margin={"l": 60, "r": 20, "t": 60, "b": 55},
            template="plotly_white",
            barmode="overlay",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return _apply_plot_axes(_figure)

    def build_visualization_figures(
        *,
        heading: str,
        input_columns: list[str],
        available_pair_count: int,
        pair_limit: int,
        histogram_limit: int,
        pair_count_widget,
        histogram_count_widget,
        histogram_bins_widget,
        pair_x_selectors,
        pair_y_selectors,
        histogram_selectors,
        scatter_series: list[dict[str, object]],
        histogram_series: list[dict[str, object]],
        histogram_reference,
        show_legends: bool,
        hover_include_name: bool = True,
        count_label: str = "Count",
    ):
        """Build the shown scatter and histogram figures for the panel."""
        _selected_pairs = []
        _seen_pairs = set()
        _pair_warnings = []
        for _position, (_x_selector, _y_selector) in enumerate(
            zip(pair_x_selectors, pair_y_selectors)
        ):
            _x_column = str(_x_selector.value)
            _y_column = str(_y_selector.value)
            if _x_column == _y_column:
                _pair_warnings.append(
                    f"Skipped {_x_column!r} versus itself."
                )
                continue
            _unordered_pair = tuple(sorted((_x_column, _y_column)))
            if _unordered_pair in _seen_pairs:
                _pair_warnings.append(
                    f"Skipped duplicate pair {_x_column!r} and "
                    f"{_y_column!r}."
                )
                continue
            _seen_pairs.add(_unordered_pair)
            _selected_pairs.append((_position, _x_column, _y_column))

        _selected_histograms = []
        _seen_histograms = set()
        _histogram_warnings = []
        for _position, _selector in enumerate(histogram_selectors):
            _column = str(_selector.value)
            if _column in _seen_histograms:
                _histogram_warnings.append(
                    f"Skipped duplicate histogram for {_column!r}."
                )
                continue
            _seen_histograms.add(_column)
            _selected_histograms.append((_position, _column))

        _scatter_figures = [
            visualization_scatter_figure(
                scatter_series,
                _x_column,
                _y_column,
                hover_include_name=hover_include_name,
            )
            for _position, _x_column, _y_column in _selected_pairs
        ]
        _histogram_figures = [
            visualization_histogram_figure(
                histogram_series,
                _column,
                int(histogram_bins_widget.value),
                reference=histogram_reference,
                hover_include_name=hover_include_name,
                count_label=count_label,
            )
            for _position, _column in _selected_histograms
        ]
        return {
            "heading": heading,
            "input_columns": input_columns,
            "available_pair_count": available_pair_count,
            "pair_limit": pair_limit,
            "histogram_limit": histogram_limit,
            "pair_count_widget": pair_count_widget,
            "histogram_count_widget": histogram_count_widget,
            "histogram_bins_widget": histogram_bins_widget,
            "pair_x_selectors": pair_x_selectors,
            "pair_y_selectors": pair_y_selectors,
            "histogram_selectors": histogram_selectors,
            "scatter_series": scatter_series,
            "histogram_series": histogram_series,
            "show_legends": show_legends,
            "selected_pairs": _selected_pairs,
            "scatter_figures": _scatter_figures,
            "pair_warnings": _pair_warnings,
            "selected_histograms": _selected_histograms,
            "histogram_figures": _histogram_figures,
            "histogram_warnings": _histogram_warnings,
        }

    def assemble_visualization_panel(
        _bundle,
        *,
        pair_save_selectors,
        histogram_save_selectors,
        pair_select_all_button,
        pair_clear_all_button,
        histogram_select_all_button,
        histogram_clear_all_button,
    ):
        """Build the tabbed panel and export selection from the figures."""

        def _selection_checkbox(_selector):
            return mo.Html(
                f"""
                <span title="Select" style="cursor: pointer;">
                    {_selector}
                </span>
                """
            ).style(
                {
                    "position": "absolute",
                    "top": "0.5rem",
                    "left": "0.5rem",
                    "z-index": "20",
                    "line-height": "1",
                }
            )

        def _shared_legend(_series, *, marker_shape: str):
            _border_radius = "50%" if marker_shape == "circle" else "2px"
            _items = "".join(
                f"""
                <span style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.4rem;
                    white-space: nowrap;
                ">
                    <span style="
                        width: 0.7rem;
                        height: 0.7rem;
                        border: 1px solid rgba(0, 0, 0, 0.55);
                        border-radius: {_border_radius};
                        background: {_item["color"]};
                        display: inline-block;
                    "></span>
                    <span>{_item["name"]}</span>
                </span>
                """
                for _item in _series
            )
            return mo.Html(
                f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 0.75rem 1.5rem;
                    padding: 0.25rem 0;
                ">
                    {_items}
                </div>
                """
            )

        heading = _bundle["heading"]
        input_columns = _bundle["input_columns"]
        available_pair_count = _bundle["available_pair_count"]
        pair_limit = _bundle["pair_limit"]
        histogram_limit = _bundle["histogram_limit"]
        pair_count_widget = _bundle["pair_count_widget"]
        histogram_count_widget = _bundle["histogram_count_widget"]
        histogram_bins_widget = _bundle["histogram_bins_widget"]
        pair_x_selectors = _bundle["pair_x_selectors"]
        pair_y_selectors = _bundle["pair_y_selectors"]
        histogram_selectors = _bundle["histogram_selectors"]
        scatter_series = _bundle["scatter_series"]
        histogram_series = _bundle["histogram_series"]
        show_legends = _bundle["show_legends"]
        _selected_pairs = _bundle["selected_pairs"]
        _scatter_figures = _bundle["scatter_figures"]
        _pair_warnings = _bundle["pair_warnings"]
        _selected_histograms = _bundle["selected_histograms"]
        _histogram_figures = _bundle["histogram_figures"]
        _histogram_warnings = _bundle["histogram_warnings"]

        _selected_scatter_plots = [
            {
                "x": _x_column,
                "y": _y_column,
                "figure": _figure,
                "include": bool(pair_save_selectors[_position].value),
            }
            for (_position, _x_column, _y_column), _figure in zip(
                _selected_pairs, _scatter_figures
            )
        ]
        _selected_histogram_plots = [
            {
                "column": _column,
                "figure": _figure,
                "include": bool(
                    histogram_save_selectors[_position].value
                ),
            }
            for (_position, _column), _figure in zip(
                _selected_histograms, _histogram_figures
            )
        ]

        _scatter_cards = [
            mo.vstack(
                [
                    _figure,
                    _selection_checkbox(pair_save_selectors[_position]),
                ],
                gap=0,
            ).style(
                {
                    "position": "relative",
                    "width": "330px",
                    "max-width": "100%",
                }
            )
            for (_position, _x_column, _y_column), _figure in zip(
                _selected_pairs, _scatter_figures
            )
        ]
        _histogram_cards = [
            mo.vstack(
                [
                    _figure,
                    _selection_checkbox(
                        histogram_save_selectors[_position]
                    ),
                ],
                gap=0,
            ).style({"position": "relative", "width": "100%"})
            for (_position, _column), _figure in zip(
                _selected_histograms, _histogram_figures
            )
        ]
        _pair_selector_rows = [
            mo.hstack(
                [_x_selector, _y_selector],
                widths="equal",
                align="start",
            )
            for _x_selector, _y_selector in zip(
                pair_x_selectors,
                pair_y_selectors,
            )
        ]
        _scatter_rows = [
            mo.hstack(
                _scatter_cards[_start : _start + 4],
                justify="start",
                align="start",
                wrap=True,
                gap=0.75,
            )
            for _start in range(0, len(_scatter_cards), 4)
        ]

        _pair_panel_items = [
            mo.md(
                f"**{available_pair_count:,} unique input pairs are "
                f"available; up to {pair_limit:,} can be shown at once.**"
            ),
            pair_count_widget,
            *_pair_selector_rows,
        ]
        if not _scatter_rows:
            _pair_panel_items.append(
                mo.md(
                    "_Increase the number of pairwise scatter plots to "
                    "begin visualizing._"
                )
            )
        if _pair_warnings:
            _pair_panel_items.append(
                mo.callout(" ".join(_pair_warnings), kind="warn")
            )
        if _scatter_rows:
            if show_legends:
                _pair_panel_items.append(
                    _shared_legend(
                        scatter_series,
                        marker_shape="circle",
                    )
                )
            _pair_panel_items.extend(
                [
                    mo.vstack(_scatter_rows, align="start", gap=0.75),
                    mo.hstack(
                        [
                            pair_select_all_button,
                            pair_clear_all_button,
                        ],
                        justify="start",
                        gap=0.5,
                    ),
                ]
            )

        _histogram_panel_items = [
            mo.md(
                f"**{len(input_columns):,} input variables are available; "
                f"up to {histogram_limit:,} histograms can be shown at "
                "once.**"
            ),
            mo.hstack(
                [histogram_count_widget, histogram_bins_widget],
                widths="equal",
                align="start",
            ),
            *histogram_selectors,
        ]
        if not _histogram_cards:
            _histogram_panel_items.append(
                mo.md(
                    "_Increase the number of marginal histograms to begin "
                    "visualizing._"
                )
            )
        if _histogram_warnings:
            _histogram_panel_items.append(
                mo.callout(" ".join(_histogram_warnings), kind="warn")
            )
        if _histogram_cards:
            if show_legends:
                _histogram_panel_items.append(
                    _shared_legend(
                        histogram_series,
                        marker_shape="square",
                    )
                )
            _histogram_panel_items.extend(
                [
                    mo.hstack(
                        _histogram_cards,
                        widths="equal",
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            histogram_select_all_button,
                            histogram_clear_all_button,
                        ],
                        justify="start",
                        gap=0.5,
                    ),
                ]
            )

        _panel = mo.vstack(
            [
                mo.md(heading),
                mo.ui.tabs(
                    {
                        "Pairwise scatter plots": mo.vstack(
                            _pair_panel_items
                        ),
                        "Marginal histograms": mo.vstack(
                            _histogram_panel_items
                        ),
                    },
                    lazy=False,
                ),
            ]
        )
        return (
            _panel,
            _selected_scatter_plots,
            _selected_histogram_plots,
        )

    def build_visualization_export_panel(
        *,
        figure_format_widget,
        scatter_plots,
        histogram_plots,
        filename_prefix: str,
        zip_basename: str | None = None,
        number_selected_only: bool = False,
    ):
        # Number filenames over included plots only, or over every shown plot.
        def _numbered(_plots, _kind, _describe):
            _source = (
                [_plot for _plot in _plots if _plot["include"]]
                if number_selected_only
                else _plots
            )
            return [
                (
                    f"{filename_prefix}_{_kind}_{_position:02d}_"
                    f"{_describe(_plot)}",
                    _plot,
                )
                for _position, _plot in enumerate(_source, start=1)
            ]

        _all_plots = [
            *_numbered(
                scatter_plots,
                "pairwise",
                lambda _plot: (
                    f"{safe_filename_component(_plot['y'])}_vs_"
                    f"{safe_filename_component(_plot['x'])}"
                ),
            ),
            *_numbered(
                histogram_plots,
                "histogram",
                lambda _plot: safe_filename_component(_plot["column"]),
            ),
        ]
        if not (scatter_plots or histogram_plots):
            return None

        _figure_format = str(figure_format_widget.value).lower()
        _selected_plots = [
            (_stem, _plot)
            for _stem, _plot in _all_plots
            if _plot["include"]
        ]

        def _figure_bytes(_plot):
            _figure = _plot["figure"]
            if _figure_format == "html":
                return _figure.to_html(
                    include_plotlyjs="cdn",
                    full_html=True,
                ).encode("utf-8")
            return _figure.to_image(
                format=_figure_format,
                scale=2 if _figure_format == "png" else 1,
            )

        if len(_selected_plots) == 1:
            _figure_stem, _selected_plot = _selected_plots[0]

            def _download_selected_figures():
                return _figure_bytes(_selected_plot)

            _download_filename = (
                f"{_figure_stem}.{_figure_format}"
            )
            _download_mimetype = FIGURE_MIME_TYPES[_figure_format]
        else:

            def _download_selected_figures():
                _buffer = io.BytesIO()
                with zipfile.ZipFile(
                    _buffer,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as _archive:
                    for _figure_stem, _selected_plot in _selected_plots:
                        _archive.writestr(
                            f"{_figure_stem}.{_figure_format}",
                            _figure_bytes(_selected_plot),
                        )
                return _buffer.getvalue()

            _download_filename = (
                f"{zip_basename or f'{filename_prefix}_figures'}.zip"
            )
            _download_mimetype = "application/zip"

        _download = mo.download(
            data=_download_selected_figures,
            filename=_download_filename,
            mimetype=_download_mimetype,
            disabled=not _selected_plots,
            label="Save selected figures…",
        )
        return mo.hstack(
            [figure_format_widget, _download],
            widths="equal",
            align="end",
        )

    return (
        assemble_visualization_panel,
        build_visualization_export_panel,
        build_visualization_figures,
        make_visualization_action_widgets,
        make_visualization_count_widgets,
        make_visualization_save_selectors,
        make_visualization_selectors,
    )


@app.cell
def _(MAX_RANDOM_SEED):
    def validate_stage1_settings(_values) -> str | None:
        try:
            _minimum = int(_values["minimum_size"])
            _maximum = int(_values["maximum_size"])
            _restarts = int(_values["restarts"])
            _calibration = int(_values["calibration_restarts"])
            _seed = int(_values["random_seed"])
        except (KeyError, TypeError, ValueError):
            return "Enter integer design sizes, restart counts, and a seed."
        if _minimum < 2:
            return "The first-stage minimum design size must be at least 2."
        if _minimum > _maximum:
            return "Minimum design size cannot exceed maximum design size."
        if _restarts < 1 or _calibration < 1:
            return "Restart counts must be positive."
        if not 0 <= _seed <= MAX_RANDOM_SEED:
            return f"Random seed must be between 0 and {MAX_RANDOM_SEED:,}."
        return None

    def validate_stage2_settings(_values) -> str | None:
        try:
            _size = int(_values["additional_size"])
            _restarts = int(_values["restarts"])
            _calibration = int(_values["calibration_restarts"])
            _seed = int(_values["random_seed"])
        except (KeyError, TypeError, ValueError):
            return "Enter an integer design size, restart counts, and a seed."
        if _size < 1:
            return "The additional design size must be positive."
        if _restarts < 1 or _calibration < 1:
            return "Restart counts must be positive."
        if not 0 <= _seed <= MAX_RANDOM_SEED:
            return f"Random seed must be between 0 and {MAX_RANDOM_SEED:,}."
        return None

    return validate_stage1_settings, validate_stage2_settings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive uniform space-filling design in 5D

    This marimo notebook serves as a UI for running a sDOE example with uniform sampling in a 5D input space. It reproduces the carbon-capture
    USF-3 workflow while making the data, column roles, scaling bounds,
    search budget, criterion, random seed, first-stage choice, and
    augmentation settings interactive.

    The workflow follows the
    [FOQUS USF-3 reference](https://foqus.readthedocs.io/en/stable/chapt_sdoe/examples-uniform.html#example-usf-3-a-uniform-space-filling-design-for-a-carbon-capture-example-in-a-5-d-input-space)
    while using the current `idaes-sdoe` API throughout.

    **Guided path:** load data → inspect the viable region → generate and
    compare Stage 1 designs → choose completed runs → add Stage 2 runs →
    export the result.

    The initial parameter suggestions follow the Jupyter notebook:

    - inputs: `G`, `lldg`, `CapturePerc`, `L`, `SteamFlow`
    - A tested starting candidate set is available at
        `examples/supporting_data/Candidate Points 8perc.csv` — select
        **Upload a candidate CSV** and upload it to get started.
    - Stage 1: minimax designs with 10, 11, and 12 runs
    - Stage 2: use the selected 12-run design as history and request 6
      additional runs
    - 10,000 random restarts and 100 calibration restarts

    Expensive work only runs after you submit settings and click a run
    button. Search results are reproducible because each stage exposes a
    random seed.
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_OUTPUT_DIR, mo):
    output_dir_widget = mo.ui.text(
        value=str(DEFAULT_OUTPUT_DIR),
        label="Session output directory",
        full_width=True,
        debounce=True,
    )
    mo.vstack(
        [
            mo.md(
                """
                Enter an absolute path or a path relative to the repository
                root. Choose a dedicated directory: files with matching
                artifact names may be replaced when a stage is rerun.
                """
            ),
            output_dir_widget,
        ]
    )
    return (output_dir_widget,)


@app.cell
def _(mo, output_dir_widget, resolve_output_directory):
    try:
        output_dir = resolve_output_directory(output_dir_widget.value)
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(
                f"Invalid output directory: {type(_exc).__name__}: {_exc}",
                kind="danger",
            ),
        )
    return (output_dir,)


@app.cell(hide_code=True)
def _(mo, output_dir):
    mo.callout(
        mo.md(
            f"""
            **Session artifact directory**

            `{output_dir}`

            Design-run artifacts use this folder. Figure downloads use your
            browser's download location.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(FIGURE_FORMATS, mo, time):
    candidate_figure_format_widget = mo.ui.dropdown(
        options=FIGURE_FORMATS,
        value="svg",
        label="Figure format",
        full_width=True,
    )
    candidate_pair_select_all_button = mo.ui.button(
        value=0,
        on_click=lambda _value: time.monotonic_ns(),
        label="Select all",
        tooltip="Select every pairwise scatter plot",
    )
    candidate_pair_clear_all_button = mo.ui.button(
        value=0,
        on_click=lambda _value: time.monotonic_ns(),
        label="Clear all",
        tooltip="Clear all pairwise scatter-plot selections",
    )
    candidate_histogram_select_all_button = mo.ui.button(
        value=0,
        on_click=lambda _value: time.monotonic_ns(),
        label="Select all",
        tooltip="Select every marginal histogram",
    )
    candidate_histogram_clear_all_button = mo.ui.button(
        value=0,
        on_click=lambda _value: time.monotonic_ns(),
        label="Clear all",
        tooltip="Clear all marginal-histogram selections",
    )
    return (
        candidate_figure_format_widget,
        candidate_histogram_clear_all_button,
        candidate_histogram_select_all_button,
        candidate_pair_clear_all_button,
        candidate_pair_select_all_button,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare the Candidate Set

    A candidate set defines the feasible operating region. Enter the
    candidate combinations in an editable table, or upload an existing CSV.
    """)
    return


@app.cell
def _(DEFAULT_INPUT_COLUMNS, mo):
    candidate_source_widget = mo.ui.radio(
        {
            "Enter a candidate table": "manual",
            "Upload a candidate CSV": "upload",
        },
        value="Enter a candidate table",
        inline=True,
        label="Candidate source",
    )
    manual_column_names_widget = mo.ui.text_area(
        value="\n".join(DEFAULT_INPUT_COLUMNS),
        label="Column names — one per line",
        rows=5,
        debounce=True,
        full_width=True,
    )
    manual_row_count_widget = mo.ui.number(
        start=2,
        stop=10_000,
        step=1,
        value=20,
        debounce=True,
        label="Number of editable rows",
    )
    candidate_upload_widget = mo.ui.file(
        filetypes=[".csv"],
        multiple=False,
        kind="area",
        label="Drop a candidate CSV here",
    )
    return (
        candidate_source_widget,
        candidate_upload_widget,
        manual_column_names_widget,
        manual_row_count_widget,
    )


@app.cell
def _(manual_column_names_widget, manual_row_count_widget, mo, pd):
    _requested_manual_columns = [
        _name.strip()
        for _name in manual_column_names_widget.value.splitlines()
        if _name.strip()
    ]
    _duplicate_manual_columns = sorted(
        {
            _name
            for _name in _requested_manual_columns
            if _requested_manual_columns.count(_name) > 1
        }
    )
    _manual_schema_errors = []
    if not _requested_manual_columns:
        _manual_schema_errors.append("Enter at least one column name.")
        _manual_columns = ["value"]
    elif _duplicate_manual_columns:
        _manual_schema_errors.append(
            "Column names must be unique: "
            + ", ".join(_duplicate_manual_columns)
        )
        _manual_columns = ["value"]
    else:
        _manual_columns = _requested_manual_columns
    _manual_row_value = manual_row_count_widget.value
    if (
        _manual_row_value is None
        or float(_manual_row_value) != int(_manual_row_value)
        or int(_manual_row_value) < 2
    ):
        _manual_schema_errors.append(
            "The number of editable rows must be an integer of at least 2."
        )
        manual_row_count = 2
    else:
        manual_row_count = int(_manual_row_value)
    manual_schema_error = (
        " ".join(_manual_schema_errors)
        if _manual_schema_errors
        else None
    )
    _manual_seed = pd.DataFrame(
        {
            _name: [None] * manual_row_count
            for _name in _manual_columns
        }
    )
    manual_candidate_editor = mo.ui.data_editor(
        _manual_seed,
        label="Candidate combinations",
        editable_columns="all",
    )
    manual_column_count = len(_manual_columns)
    return (
        manual_candidate_editor,
        manual_column_count,
        manual_row_count,
        manual_schema_error,
    )


@app.cell
def _(
    candidate_source_widget,
    candidate_upload_widget,
    manual_candidate_editor,
    manual_column_count,
    manual_column_names_widget,
    manual_row_count,
    manual_row_count_widget,
    manual_schema_error,
    mo,
):
    _manual_schema_message = (
        mo.callout(manual_schema_error, kind="danger")
        if manual_schema_error
        else mo.md(
            "_Set the headers and row count before entering values; changing "
            "the schema resets the grid. Blank rows are ignored. For very "
            "large tables, CSV upload is more practical._"
        )
    )
    _candidate_source_panel = (
        mo.vstack(
            [
                mo.hstack(
                    [
                        manual_column_names_widget,
                        manual_row_count_widget,
                    ],
                    widths=[3, 1],
                    align="start",
                ),
                _manual_schema_message,
                mo.md(
                    f"**Grid size:** {manual_row_count:,} rows × "
                    f"{manual_column_count:,} columns"
                ),
                manual_candidate_editor,
            ]
        )
        if candidate_source_widget.value == "manual"
        else candidate_upload_widget
    )
    mo.vstack([candidate_source_widget, _candidate_source_panel])
    return


@app.cell
def _(
    candidate_source_widget,
    candidate_upload_widget,
    clean_manual_candidate,
    manual_candidate_editor,
    manual_schema_error,
    mo,
    read_uploaded_csv,
):
    try:
        if candidate_source_widget.value == "manual":
            if manual_schema_error:
                raise ValueError(manual_schema_error)
            _candidate_frame = clean_manual_candidate(
                manual_candidate_editor.value
            )
            _candidate_label = "Manually entered candidate table"
        else:
            if not candidate_upload_widget.value:
                raise ValueError("Upload a candidate CSV to continue.")
            _candidate_upload = candidate_upload_widget.value[0]
            _candidate_frame = read_uploaded_csv(_candidate_upload)
            _candidate_label = _candidate_upload.name
        if _candidate_frame.empty:
            raise ValueError(
                "The candidate table contains no completed rows."
            )
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(
                f"Could not load candidate data: {type(_exc).__name__}: {_exc}",
                kind="danger",
            ),
        )
    candidate_data = _candidate_frame
    candidate_name = _candidate_label
    return candidate_data, candidate_name


@app.cell
def _(AUTO_INDEX, DEFAULT_INPUT_COLUMNS, candidate_data, mo):
    _numeric_columns = [
        _name
        for _name in candidate_data.columns
        if candidate_data[_name].dtype.kind in "biufc"
    ]
    _default_inputs = [
        _name for _name in DEFAULT_INPUT_COLUMNS if _name in _numeric_columns
    ]
    if not _default_inputs:
        _default_inputs = [
            _name
            for _name in _numeric_columns
            if _name not in {"Test No.", "CO2 captured", "__id"}
        ]
    _default_index_label = (
        "__id" if "__id" in _numeric_columns else "Generate __id"
    )
    _index_options = {"Generate __id": AUTO_INDEX}
    _index_options.update(
        {_name: _name for _name in _numeric_columns}
    )
    input_columns_widget = mo.ui.multiselect(
        options=_numeric_columns,
        value=_default_inputs,
        label="Input columns used in distances",
        full_width=True,
    )
    index_column_widget = mo.ui.dropdown(
        options=_index_options,
        value=_default_index_label,
        label="Index column",
        full_width=True,
    )
    mo.hstack(
        [input_columns_widget, index_column_widget],
        widths=[2, 1],
        align="start",
    )
    return index_column_widget, input_columns_widget


@app.cell
def _(
    AUTO_INDEX,
    candidate_data,
    index_column_widget,
    input_columns_widget,
    mo,
    validate_candidate,
):
    input_columns = list(input_columns_widget.value)
    index_column = (
        None
        if index_column_widget.value == AUTO_INDEX
        else str(index_column_widget.value)
    )
    try:
        validate_candidate(candidate_data, input_columns, index_column)
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(
                f"Column-role validation failed: "
                f"{type(_exc).__name__}: {_exc}",
                kind="danger",
            ),
        )
    return index_column, input_columns


@app.cell
def _(DEFAULT_HISTOGRAM_BINS, candidate_data, input_columns, mo, pd):
    _bounds_seed = pd.DataFrame(
        {
            "input": input_columns,
            "minimum": [
                float(candidate_data[_name].min())
                for _name in input_columns
            ],
            "maximum": [
                float(candidate_data[_name].max())
                for _name in input_columns
            ],
        }
    )
    bounds_editor = mo.ui.data_editor(
        _bounds_seed,
        label="Distance-scaling bounds",
        editable_columns=["minimum", "maximum"],
    )
    histogram_bins_widget = mo.ui.slider(
        start=5,
        stop=50,
        step=1,
        value=DEFAULT_HISTOGRAM_BINS,
        show_value=True,
        include_input=True,
        label="Target histogram bins",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                Edit bounds to change the relative emphasis of each input in
                distance calculations. Bounds scale distances; they do not
                filter candidate rows. Saved parity plots use all active inputs
                and {DEFAULT_HISTOGRAM_BINS} target bins, matching the Jupyter
                example.
                """
            ),
            bounds_editor,
        ]
    )
    return bounds_editor, histogram_bins_widget


@app.cell
def _(
    ColumnRoles,
    bounds_editor,
    bounds_from_editor,
    candidate_data,
    index_column,
    input_columns,
    mo,
    prepare_design_setup,
):
    try:
        design_bounds = bounds_from_editor(bounds_editor.value, input_columns)
        candidate_setup = prepare_design_setup(
            candidate=candidate_data,
            roles=ColumnRoles(
                index=index_column,
                inputs=input_columns,
            ),
            bounds=design_bounds,
            auto_index=index_column is None,
            index_column="__id",
        )
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(
                f"Could not prepare the design setup: "
                f"{type(_exc).__name__}: {_exc}",
                kind="danger",
            ),
        )
    return (candidate_setup,)


@app.cell
def _(candidate_data, candidate_name, candidate_setup, input_columns, mo, pd):
    _missing = int(candidate_data[input_columns].isna().sum().sum())
    _duplicate_inputs = int(
        candidate_data.duplicated(subset=input_columns).sum()
    )
    _roles = pd.DataFrame(
        {
            "column": candidate_setup.candidate.columns,
            "role": [
                (
                    "Index"
                    if _name == candidate_setup.roles.index
                    else "Input"
                    if _name in input_columns
                    else "Excluded (kept in exports)"
                )
                for _name in candidate_setup.candidate.columns
            ],
        }
    )
    _ranges = candidate_setup.candidate[input_columns].agg(["min", "max"]).T
    _ranges.index.name = "input"
    _ranges = _ranges.reset_index()
    _candidate_table = mo.ui.table(
        candidate_setup.candidate,
        selection=None,
        pagination=True,
        page_size=10,
        show_column_summaries=True,
        show_download=True,
        label="Candidate rows",
    )
    _roles_table = mo.ui.table(
        _roles,
        selection=None,
        pagination=False,
        show_download=False,
    )
    _ranges_table = mo.ui.table(
        _ranges,
        selection=None,
        pagination=False,
        show_download=False,
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        len(candidate_setup.candidate),
                        label="Candidate rows",
                        bordered=True,
                    ),
                    mo.stat(
                        len(candidate_setup.candidate.columns),
                        label="Columns",
                        bordered=True,
                    ),
                    mo.stat(
                        len(input_columns),
                        label="Active inputs",
                        bordered=True,
                    ),
                    mo.stat(
                        _missing,
                        label="Missing input values",
                        bordered=True,
                    ),
                    mo.stat(
                        _duplicate_inputs,
                        label="Duplicate input rows",
                        bordered=True,
                    ),
                ],
                widths="equal",
                wrap=True,
            ),
            mo.md(f"**Loaded:** `{candidate_name}`"),
            mo.ui.tabs(
                {
                    "Candidate table": _candidate_table,
                    "Column roles": _roles_table,
                    "Input ranges": _ranges_table,
                },
                lazy=True,
            ),
        ]
    )
    return


@app.cell
def _(MAX_HISTOGRAMS, MAX_SCATTER_PLOTS, input_columns, mo):
    candidate_available_pair_count = (
        len(input_columns) * (len(input_columns) - 1) // 2
    )
    candidate_pair_plot_limit = min(
        candidate_available_pair_count,
        MAX_SCATTER_PLOTS,
    )
    candidate_histogram_limit = min(
        len(input_columns),
        MAX_HISTOGRAMS,
    )
    candidate_pair_plot_count_widget = mo.ui.number(
        start=0,
        stop=max(1, candidate_pair_plot_limit),
        step=1,
        value=0,
        debounce=True,
        label="Number of pairwise scatter plots",
        disabled=candidate_pair_plot_limit == 0,
    )
    candidate_histogram_count_widget = mo.ui.number(
        start=0,
        stop=max(1, candidate_histogram_limit),
        step=1,
        value=0,
        debounce=True,
        label="Number of marginal histograms",
        disabled=candidate_histogram_limit == 0,
    )
    return (
        candidate_available_pair_count,
        candidate_histogram_count_widget,
        candidate_histogram_limit,
        candidate_pair_plot_count_widget,
        candidate_pair_plot_limit,
    )


@app.cell
def _(
    candidate_histogram_count_widget,
    candidate_pair_plot_count_widget,
    input_columns,
    make_visualization_selectors,
):
    (
        candidate_pair_x_selectors,
        candidate_pair_y_selectors,
        candidate_histogram_selectors,
    ) = make_visualization_selectors(
        input_columns,
        int(candidate_pair_plot_count_widget.value or 0),
        int(candidate_histogram_count_widget.value or 0),
    )
    return (
        candidate_histogram_selectors,
        candidate_pair_x_selectors,
        candidate_pair_y_selectors,
    )


@app.cell
def _(
    candidate_pair_clear_all_button,
    candidate_pair_plot_count_widget,
    candidate_pair_select_all_button,
    make_visualization_save_selectors,
):
    candidate_pair_save_selectors = make_visualization_save_selectors(
        int(candidate_pair_plot_count_widget.value or 0),
        candidate_pair_select_all_button,
        candidate_pair_clear_all_button,
    )
    return (candidate_pair_save_selectors,)


@app.cell
def _(
    candidate_histogram_clear_all_button,
    candidate_histogram_count_widget,
    candidate_histogram_select_all_button,
    make_visualization_save_selectors,
):
    candidate_histogram_save_selectors = (
        make_visualization_save_selectors(
            int(candidate_histogram_count_widget.value or 0),
            candidate_histogram_select_all_button,
            candidate_histogram_clear_all_button,
        )
    )
    return (candidate_histogram_save_selectors,)


@app.cell
def _(
    build_visualization_figures,
    candidate_available_pair_count,
    candidate_histogram_count_widget,
    candidate_histogram_limit,
    candidate_histogram_selectors,
    candidate_pair_plot_count_widget,
    candidate_pair_plot_limit,
    candidate_pair_x_selectors,
    candidate_pair_y_selectors,
    candidate_setup,
    histogram_bins_widget,
    input_columns,
):
    candidate_visualization_figures = build_visualization_figures(
        heading="### Candidate Sampling Space Visualization",
        input_columns=input_columns,
        available_pair_count=candidate_available_pair_count,
        pair_limit=candidate_pair_plot_limit,
        histogram_limit=candidate_histogram_limit,
        pair_count_widget=candidate_pair_plot_count_widget,
        histogram_count_widget=candidate_histogram_count_widget,
        histogram_bins_widget=histogram_bins_widget,
        pair_x_selectors=candidate_pair_x_selectors,
        pair_y_selectors=candidate_pair_y_selectors,
        histogram_selectors=candidate_histogram_selectors,
        scatter_series=[
            {
                "data": candidate_setup.candidate,
                "name": "Candidate points",
                "color": "rgba(51, 51, 51, 0.62)",
                "size": 7,
            }
        ],
        histogram_series=[
            {
                "data": candidate_setup.candidate,
                "name": "Candidate points",
                "color": "rgba(76, 120, 168, 0.78)",
                "opacity": 1.0,
            }
        ],
        histogram_reference=candidate_setup.candidate,
        show_legends=False,
        hover_include_name=False,
        count_label="Candidate count",
    )
    return (candidate_visualization_figures,)


@app.cell
def _(
    assemble_visualization_panel,
    candidate_histogram_clear_all_button,
    candidate_histogram_save_selectors,
    candidate_histogram_select_all_button,
    candidate_pair_clear_all_button,
    candidate_pair_save_selectors,
    candidate_pair_select_all_button,
    candidate_visualization_figures,
):
    (
        _candidate_visualization,
        candidate_selected_scatter_plots,
        candidate_selected_histograms,
    ) = assemble_visualization_panel(
        candidate_visualization_figures,
        pair_save_selectors=candidate_pair_save_selectors,
        histogram_save_selectors=candidate_histogram_save_selectors,
        pair_select_all_button=candidate_pair_select_all_button,
        pair_clear_all_button=candidate_pair_clear_all_button,
        histogram_select_all_button=candidate_histogram_select_all_button,
        histogram_clear_all_button=candidate_histogram_clear_all_button,
    )
    _candidate_visualization
    return candidate_selected_histograms, candidate_selected_scatter_plots


@app.cell(hide_code=True)
def _(
    build_visualization_export_panel,
    candidate_figure_format_widget,
    candidate_selected_histograms,
    candidate_selected_scatter_plots,
):
    build_visualization_export_panel(
        figure_format_widget=candidate_figure_format_widget,
        scatter_plots=candidate_selected_scatter_plots,
        histogram_plots=candidate_selected_histograms,
        filename_prefix="candidate",
        zip_basename="candidate_sampling_space_figures",
        number_selected_only=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Stage 1 — Design and Execution

    Configure the design criterion, design-size range, number of random
    restarts, runtime-calibration restarts, and random seed. Apply the
    settings to enable runtime estimation and generate one design for each
    integer size in the selected range.
    """)
    return


@app.cell
def _(
    candidate_setup,
    make_common_stage_widgets,
    mo,
    validate_stage1_settings,
):
    _candidate_count = len(candidate_setup.candidate)
    _default_minimum = min(10, _candidate_count)
    _default_maximum = min(12, _candidate_count)
    _stage1_batch = mo.md(
        """
        | Setting | Value |
        |:--|:--|
        | Criterion | {criterion} |
        | Minimum design size | {minimum_size} |
        | Maximum design size | {maximum_size} |
        | Random restarts | {restarts} |
        | Calibration restarts | {calibration_restarts} |
        | Random seed | {random_seed} |
        """
    ).batch(
        minimum_size=mo.ui.number(
            start=2,
            stop=_candidate_count,
            step=1,
            value=_default_minimum,
        ),
        maximum_size=mo.ui.number(
            start=2,
            stop=_candidate_count,
            step=1,
            value=_default_maximum,
        ),
        **make_common_stage_widgets(2026),
    )
    stage1_settings_form = _stage1_batch.form(
        submit_button_label="Apply Stage 1 settings",
        validate=validate_stage1_settings,
        clear_on_submit=False,
    )
    stage1_settings_form
    return (stage1_settings_form,)


@app.cell
def _(candidate_setup, mo, stage1_settings_form):
    mo.stop(
        stage1_settings_form.value is None,
        mo.callout(
            "Apply the Stage 1 settings to enable estimation and search.",
            kind="info",
        ),
    )
    _stage1_values = stage1_settings_form.value
    stage1_config = {
        "mode": str(_stage1_values["criterion"]),
        "minimum_size": int(_stage1_values["minimum_size"]),
        "maximum_size": int(_stage1_values["maximum_size"]),
        "design_sizes": list(
            range(
                int(_stage1_values["minimum_size"]),
                int(_stage1_values["maximum_size"]) + 1,
            )
        ),
        "num_restarts": int(_stage1_values["restarts"]),
        "calibration_restarts": int(
            _stage1_values["calibration_restarts"]
        ),
        "random_seed": int(_stage1_values["random_seed"]),
    }
    mo.stop(
        stage1_config["maximum_size"] > len(candidate_setup.candidate),
        mo.callout(
            "The maximum design size exceeds the candidate count.",
            kind="danger",
        ),
    )
    return (stage1_config,)


@app.cell
def _(mo, stage1_config):
    stage1_estimate_button = mo.ui.run_button(
        label="Estimate Stage 1 runtime",
        kind="neutral",
        tooltip="Calibrate a projected runtime without running the full search.",
    )
    stage1_run_button = mo.ui.run_button(
        label="Run Stage 1",
        kind="success",
        tooltip="Run and save every requested design size.",
    )
    _restart_note = (
        mo.callout(
            "This setting can be expensive. Estimate runtime before running.",
            kind="warn",
        )
        if stage1_config["num_restarts"] >= 100_000
        else mo.md(
            f"Search budget: **{stage1_config['num_restarts']:,} restarts "
            "per design size**."
        )
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"""
                    **Applied Stage 1 settings**

                    `{stage1_config['mode']}` · sizes
                    `{stage1_config['minimum_size']}–{stage1_config['maximum_size']}`
                    · `{stage1_config['num_restarts']:,}` restarts/design ·
                    `{stage1_config['calibration_restarts']:,}` calibration
                    restarts · seed `{stage1_config['random_seed']}`

                    The buttons below use this submitted snapshot. Apply the
                    form again after changing any visible setting.
                    """
                ),
                kind="info",
            ),
            _restart_note,
            mo.hstack(
                [stage1_estimate_button, stage1_run_button],
                justify="start",
            ),
        ]
    )
    return stage1_estimate_button, stage1_run_button


@app.cell
def _(
    candidate_setup,
    estimate_uniform_runtime,
    format_duration,
    mo,
    pd,
    stage1_config,
    stage1_estimate_button,
):
    mo.stop(not stage1_estimate_button.value)
    with mo.status.spinner(
        title="Calibrating Stage 1 runtime",
        subtitle=(
            f"{stage1_config['calibration_restarts']:,} calibration restarts "
            "per size"
        ),
    ):
        _stage1_estimate = estimate_uniform_runtime(
            setup=candidate_setup,
            design_sizes=stage1_config["design_sizes"],
            target_restarts=stage1_config["num_restarts"],
            mode=stage1_config["mode"],
            calibration_restarts=stage1_config["calibration_restarts"],
            random_state=stage1_config["random_seed"],
        )
    _stage1_estimate_table = pd.DataFrame(
        [
            {
                "design_size": int(_size),
                "estimated_seconds": float(_seconds),
                "estimated_time": format_duration(float(_seconds)),
            }
            for _size, _seconds in _stage1_estimate[
                "by_design_size"
            ].items()
        ]
    )
    mo.vstack(
        [
            mo.callout(
                "Estimated total: "
                + format_duration(_stage1_estimate["total_seconds"]),
                kind="info",
            ),
            mo.ui.table(
                _stage1_estimate_table,
                selection=None,
                pagination=False,
                show_download=False,
            ),
            mo.md(
                "_This is a linear projection from a short calibration run; "
                "actual time depends on the machine and current load._"
            ),
        ]
    )
    return


@app.cell
def _(
    DEFAULT_HISTOGRAM_BINS,
    candidate_setup,
    create_run_dirs,
    design_uniform_batch,
    mo,
    new_run_stamp,
    output_dir,
    pair_figure,
    pd,
    save_figure_artifact,
    stage1_config,
    stage1_run_button,
    time,
    write_csv,
    write_json,
    write_runtime_estimate,
):
    mo.stop(not stage1_run_button.value)
    _stage1_run_stamp = new_run_stamp()
    _stage1_dir, _stage1_design_dir, _stage1_plot_dir = create_run_dirs(
        output_dir, "stage1", stage1_config["mode"], _stage1_run_stamp
    )
    _stage1_warnings: list[str] = []

    with mo.status.spinner(
        title="Running Stage 1",
        subtitle=(
            f"{len(stage1_config['design_sizes'])} design(s), "
            f"{stage1_config['num_restarts']:,} restarts each"
        ),
        remove_on_exit=False,
    ):
        write_csv(
            candidate_setup.candidate,
            output_dir / "candidate_set.csv",
        )
        _candidate_export_figure = pair_figure(
            candidate_setup.candidate,
            candidate_setup.roles.inputs,
            title="USF-3 candidate set pairwise plot",
            target_bins=DEFAULT_HISTOGRAM_BINS,
        )
        save_figure_artifact(
            _candidate_export_figure,
            output_dir / "candidate_pairwise.pdf",
            _stage1_warnings,
        )

        _stage1_estimate_payload = write_runtime_estimate(
            candidate_setup,
            stage1_config["design_sizes"],
            stage1_config,
            _stage1_dir,
        )

        _stage1_started = time.perf_counter()
        _stage1_results = design_uniform_batch(
            setup=candidate_setup,
            design_sizes=stage1_config["design_sizes"],
            num_restarts=stage1_config["num_restarts"],
            mode=stage1_config["mode"],
            random_state=stage1_config["random_seed"],
        )
        _stage1_elapsed = time.perf_counter() - _stage1_started

        _stage1_rows: list[dict[str, object]] = []
        _stage1_results_by_size = {}
        for _result in _stage1_results:
            _stage1_results_by_size[_result.design_size] = _result
            _design_path = (
                _stage1_design_dir
                / (
                    f"stage1_{stage1_config['mode']}_design_"
                    f"{_result.design_size:02d}.csv"
                )
            )
            _plot_path = (
                _stage1_plot_dir
                / (
                    f"stage1_{stage1_config['mode']}_design_"
                    f"{_result.design_size:02d}_pairwise.pdf"
                )
            )
            write_csv(_result.design, _design_path)
            _figure = pair_figure(
                _result.design,
                candidate_setup.roles.inputs,
                candidate=candidate_setup.candidate,
                title=(
                    f"Stage 1 {stage1_config['mode']} design, "
                    f"size {_result.design_size}, "
                    f"criterion={_result.criterion_value:.6f}"
                ),
                target_bins=DEFAULT_HISTOGRAM_BINS,
            )
            _saved_plot_relative = save_figure_artifact(
                _figure,
                _plot_path,
                _stage1_warnings,
                relative_to=output_dir,
            )
            _stage1_rows.append(
                {
                    "mode": _result.mode,
                    "design_size": _result.design_size,
                    "num_restarts": _result.num_restarts,
                    "criterion_value": _result.criterion_value,
                    "elapsed_seconds": _result.elapsed_time,
                    "design_file": str(_design_path.relative_to(output_dir)),
                    "pair_plot_artifact": _saved_plot_relative,
                }
            )

        _stage1_summary = pd.DataFrame(_stage1_rows)
        write_csv(
            _stage1_summary,
            _stage1_dir / "created_designs_summary.csv",
        )
        write_csv(
            _stage1_summary,
            output_dir / "stage1_created_designs_summary.csv",
        )
        write_json(
            {
                **stage1_config,
                "elapsed_seconds": _stage1_elapsed,
                "bounds": candidate_setup.bounds,
                "input_columns": candidate_setup.roles.inputs,
                "index_column": candidate_setup.roles.index,
            },
            _stage1_dir / "run_summary.json",
        )

    stage1_run = {
        "config": dict(stage1_config),
        "setup": candidate_setup,
        "estimate": _stage1_estimate_payload,
        "results_by_size": _stage1_results_by_size,
        "summary": _stage1_summary,
        "elapsed_seconds": _stage1_elapsed,
        "directory": _stage1_dir,
        "warnings": _stage1_warnings,
    }
    _stage1_message = (
        f"Created {len(_stage1_results)} Stage 1 design(s) in "
        f"{_stage1_elapsed:.2f} seconds. Results were saved under "
        f"{_stage1_dir}."
    )
    mo.callout(
        _stage1_message
        + (
            "\n\nStatic export warnings:\n- "
            + "\n- ".join(_stage1_warnings)
            if _stage1_warnings
            else ""
        ),
        kind="success" if not _stage1_warnings else "warn",
    )
    return (stage1_run,)


@app.cell
def _(mo, stage1_run):
    _stage1_sizes = sorted(stage1_run["results_by_size"])
    _stage1_default_size = 12 if 12 in _stage1_sizes else max(_stage1_sizes)
    _stage1_picker_options = {
        (
            f"{_size} runs — criterion "
            f"{stage1_run['results_by_size'][_size].criterion_value:.6g}"
        ): _size
        for _size in _stage1_sizes
    }
    _stage1_default_label = next(
        _label
        for _label, _size in _stage1_picker_options.items()
        if _size == _stage1_default_size
    )
    stage1_design_picker = mo.ui.dropdown(
        options=_stage1_picker_options,
        value=_stage1_default_label,
        label="Stage 1 design to inspect and carry forward",
        full_width=True,
    )
    _stage1_summary_table = mo.ui.table(
        stage1_run["summary"],
        selection=None,
        pagination=False,
        show_column_summaries=True,
        show_download=True,
    )
    mo.vstack(
        [
            mo.md("### Inspect Stage 1 designs"),
            _stage1_summary_table,
            stage1_design_picker,
        ]
    )
    return (stage1_design_picker,)


@app.cell
def _(
    csv_bytes,
    mo,
    output_dir,
    stage1_design_picker,
    stage1_run,
    zip_directory,
):
    _selected_size = int(stage1_design_picker.value)
    _selected_result = stage1_run["results_by_size"][_selected_size]
    _selected_table = mo.ui.table(
        _selected_result.design,
        selection=None,
        pagination=True,
        page_size=12,
        show_column_summaries=True,
        show_download=True,
    )
    _selected_download = mo.download(
        data=lambda: csv_bytes(_selected_result.design),
        filename=(
            f"stage1_{stage1_run['config']['mode']}_"
            f"design_{_selected_size:02d}.csv"
        ),
        mimetype="text/csv",
        label="Download selected Stage 1 design",
    )
    _stage1_archive = mo.download(
        data=lambda: zip_directory(output_dir),
        filename=f"{output_dir.name}.zip",
        mimetype="application/zip",
        label="Download all session artifacts",
    )
    stage1_selection = {
        "size": _selected_size,
        "result": _selected_result,
        "design": _selected_result.design.copy(),
    }
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        _selected_size,
                        label="Selected runs",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_selected_result.criterion_value:.6g}",
                        label="Criterion",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_selected_result.elapsed_time:.2f} s",
                        label="Search time",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            _selected_table,
            mo.hstack(
                [_selected_download, _stage1_archive],
                justify="start",
            ),
        ]
    )
    return (stage1_selection,)


@app.cell
def _(make_visualization_action_widgets):
    (
        stage1_figure_format_widget,
        stage1_pair_select_all_button,
        stage1_pair_clear_all_button,
        stage1_histogram_select_all_button,
        stage1_histogram_clear_all_button,
    ) = make_visualization_action_widgets("Stage 1")
    return (
        stage1_figure_format_widget,
        stage1_histogram_clear_all_button,
        stage1_histogram_select_all_button,
        stage1_pair_clear_all_button,
        stage1_pair_select_all_button,
    )


@app.cell
def _(input_columns, make_visualization_count_widgets):
    (
        stage1_available_pair_count,
        stage1_pair_plot_limit,
        stage1_histogram_limit,
        stage1_pair_plot_count_widget,
        stage1_histogram_count_widget,
    ) = make_visualization_count_widgets(input_columns, "Stage 1")
    return (
        stage1_available_pair_count,
        stage1_histogram_count_widget,
        stage1_histogram_limit,
        stage1_pair_plot_count_widget,
        stage1_pair_plot_limit,
    )


@app.cell
def _(
    input_columns,
    make_visualization_selectors,
    stage1_histogram_count_widget,
    stage1_pair_plot_count_widget,
):
    (
        stage1_pair_x_selectors,
        stage1_pair_y_selectors,
        stage1_histogram_selectors,
    ) = make_visualization_selectors(
        input_columns,
        int(stage1_pair_plot_count_widget.value or 0),
        int(stage1_histogram_count_widget.value or 0),
    )
    return (
        stage1_histogram_selectors,
        stage1_pair_x_selectors,
        stage1_pair_y_selectors,
    )


@app.cell
def _(
    make_visualization_save_selectors,
    stage1_histogram_clear_all_button,
    stage1_histogram_count_widget,
    stage1_histogram_select_all_button,
    stage1_pair_clear_all_button,
    stage1_pair_plot_count_widget,
    stage1_pair_select_all_button,
):
    stage1_pair_save_selectors = make_visualization_save_selectors(
        int(stage1_pair_plot_count_widget.value or 0),
        stage1_pair_select_all_button,
        stage1_pair_clear_all_button,
    )
    stage1_histogram_save_selectors = make_visualization_save_selectors(
        int(stage1_histogram_count_widget.value or 0),
        stage1_histogram_select_all_button,
        stage1_histogram_clear_all_button,
    )
    return stage1_histogram_save_selectors, stage1_pair_save_selectors


@app.cell
def _(
    CANDIDATE_POINT_COLOR,
    DESIGN_POINT_COLOR,
    build_visualization_figures,
    histogram_bins_widget,
    input_columns,
    stage1_available_pair_count,
    stage1_histogram_count_widget,
    stage1_histogram_limit,
    stage1_histogram_selectors,
    stage1_pair_plot_count_widget,
    stage1_pair_plot_limit,
    stage1_pair_x_selectors,
    stage1_pair_y_selectors,
    stage1_run,
    stage1_selection,
):
    stage1_visualization_figures = build_visualization_figures(
        heading="### Stage 1 Design Visualization",
        input_columns=input_columns,
        available_pair_count=stage1_available_pair_count,
        pair_limit=stage1_pair_plot_limit,
        histogram_limit=stage1_histogram_limit,
        pair_count_widget=stage1_pair_plot_count_widget,
        histogram_count_widget=stage1_histogram_count_widget,
        histogram_bins_widget=histogram_bins_widget,
        pair_x_selectors=stage1_pair_x_selectors,
        pair_y_selectors=stage1_pair_y_selectors,
        histogram_selectors=stage1_histogram_selectors,
        scatter_series=[
            {
                "data": stage1_run["setup"].candidate,
                "name": "Candidate points",
                "color": CANDIDATE_POINT_COLOR,
                "size": 5,
            },
            {
                "data": stage1_selection["design"],
                "name": "Design points",
                "color": DESIGN_POINT_COLOR,
                "size": 6,
            },
        ],
        histogram_series=[
            {
                "data": stage1_selection["design"],
                "name": "Design points",
                "color": DESIGN_POINT_COLOR,
                "opacity": 0.75,
            }
        ],
        histogram_reference=stage1_run["setup"].candidate,
        show_legends=True,
    )
    return (stage1_visualization_figures,)


@app.cell
def _(
    assemble_visualization_panel,
    stage1_histogram_clear_all_button,
    stage1_histogram_save_selectors,
    stage1_histogram_select_all_button,
    stage1_pair_clear_all_button,
    stage1_pair_save_selectors,
    stage1_pair_select_all_button,
    stage1_visualization_figures,
):
    (
        _stage1_visualization,
        stage1_selected_scatter_plots,
        stage1_selected_histogram_plots,
    ) = assemble_visualization_panel(
        stage1_visualization_figures,
        pair_save_selectors=stage1_pair_save_selectors,
        histogram_save_selectors=stage1_histogram_save_selectors,
        pair_select_all_button=stage1_pair_select_all_button,
        pair_clear_all_button=stage1_pair_clear_all_button,
        histogram_select_all_button=stage1_histogram_select_all_button,
        histogram_clear_all_button=stage1_histogram_clear_all_button,
    )
    _stage1_visualization
    return stage1_selected_histogram_plots, stage1_selected_scatter_plots


@app.cell(hide_code=True)
def _(
    build_visualization_export_panel,
    stage1_figure_format_widget,
    stage1_run,
    stage1_selected_histogram_plots,
    stage1_selected_scatter_plots,
    stage1_selection,
):
    build_visualization_export_panel(
        figure_format_widget=stage1_figure_format_widget,
        scatter_plots=stage1_selected_scatter_plots,
        histogram_plots=stage1_selected_histogram_plots,
        filename_prefix=(
            f"stage1_{stage1_run['config']['mode']}_design_"
            f"{stage1_selection['size']:02d}"
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Stage 2 — Design and Execution
    ### Previous Data Selection
    By default, the selected Stage 1 design is treated as completed **previous data**.
    You can instead upload and aggregate one or more CSV files containing
    actual completed runs.

    Final results always report overlap and unique-run counts.
    """)
    return


@app.cell
def _(mo):
    previous_source_widget = mo.ui.radio(
        {
            "Use selected Stage 1 design": "stage1",
            "Upload completed-run CSV file(s)": "upload",
        },
        value="Use selected Stage 1 design",
        inline=True,
        label="Previous-data source",
    )
    previous_upload_widget = mo.ui.file(
        filetypes=[".csv"],
        multiple=True,
        kind="area",
        label="Drop one or more previous-data CSVs here",
    )
    exclude_previous_widget = mo.ui.switch(
        value=True,
        label="Prevent repeated runs",
    )
    return (
        exclude_previous_widget,
        previous_source_widget,
        previous_upload_widget,
    )


@app.cell
def _(
    exclude_previous_widget,
    mo,
    previous_source_widget,
    previous_upload_widget,
    stage1_selection,
):
    mo.vstack(
        [
            mo.md(
                f"The current Stage 1 selection contains "
                f"**{stage1_selection['size']} rows**."
            ),
            previous_source_widget,
            previous_upload_widget,
            exclude_previous_widget,
            mo.md(
                "_On: remove candidates matching previous input settings. "
                "Off: keep all candidates eligible, so repeated runs are "
                "possible._"
            ),
        ]
    )
    return


@app.cell
def _(
    ColumnRoles,
    align_previous_frame,
    exclude_previous_candidates,
    exclude_previous_widget,
    mo,
    pd,
    prepare_design_setup,
    previous_source_widget,
    previous_upload_widget,
    read_uploaded_csv,
    stage1_run,
    stage1_selection,
):
    _stage2_base_candidate = stage1_run["setup"].candidate.copy()
    _stage2_roles = ColumnRoles(
        index=stage1_run["setup"].roles.index,
        inputs=list(stage1_run["setup"].roles.inputs),
    )
    _previous_notes: list[str] = []
    try:
        if previous_source_widget.value == "stage1":
            _stage2_previous = stage1_selection["design"].copy()
            _trust_previous_index = True
            _previous_label = (
                f"Selected Stage 1 design ({len(_stage2_previous)} rows)"
            )
        else:
            if not previous_upload_widget.value:
                raise ValueError(
                    "Upload at least one previous-data CSV to continue."
                )
            _previous_frames = [
                read_uploaded_csv(_upload)
                for _upload in previous_upload_widget.value
            ]
            _combined_previous = (
                pd.concat(
                    _previous_frames,
                    join="outer",
                    ignore_index=True,
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )
            _stage2_previous, _previous_notes = align_previous_frame(
                _combined_previous,
                _stage2_base_candidate,
                _stage2_roles,
            )
            _trust_previous_index = False
            _previous_label = (
                f"{len(_previous_frames)} uploaded file(s), "
                f"{len(_stage2_previous)} aggregated rows"
            )

        if exclude_previous_widget.value:
            _stage2_candidate, _overlap_details = (
                exclude_previous_candidates(
                    _stage2_base_candidate,
                    _stage2_previous,
                    _stage2_roles,
                    trust_index=_trust_previous_index,
                )
            )
        else:
            _stage2_candidate = _stage2_base_candidate.copy()
            _overlap_details = {
                "removed": 0,
                "matched_by_inputs": 0,
                "matched_by_index": 0,
            }
        if _stage2_candidate.empty:
            raise ValueError(
                "No candidates remain after excluding previous runs."
            )
        _stage2_input_bounds = {
            _name: stage1_run["setup"].bounds[_name]
            for _name in _stage2_roles.inputs
        }
        _stage2_setup = prepare_design_setup(
            candidate=_stage2_candidate,
            previous=_stage2_previous,
            roles=_stage2_roles,
            bounds=_stage2_input_bounds,
            auto_index=False,
        )
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(
                f"Could not prepare previous data: {type(_exc).__name__}: {_exc}",
                kind="danger",
            ),
        )

    stage2_context = {
        "setup": _stage2_setup,
        "base_candidate": _stage2_base_candidate,
        "previous": _stage2_previous,
        "previous_source": str(previous_source_widget.value),
        "previous_label": _previous_label,
        "previous_notes": _previous_notes,
        "exclude_previous": bool(exclude_previous_widget.value),
        "overlap_details": _overlap_details,
    }
    return (stage2_context,)


@app.cell
def _(mo, stage2_context):
    _overlap_details = stage2_context["overlap_details"]
    _stage2_notes = [
        mo.stat(
            len(stage2_context["previous"]),
            label="Previous rows",
            bordered=True,
        ),
        mo.stat(
            len(stage2_context["setup"].candidate),
            label="Available candidates",
            bordered=True,
        ),
        mo.stat(
            _overlap_details["removed"],
            label="Prior candidates removed",
            bordered=True,
        ),
    ]
    _note_blocks = [
        mo.callout(_note, kind="info")
        for _note in stage2_context["previous_notes"]
    ]
    if not stage2_context["exclude_previous"]:
        _note_blocks.append(
            mo.callout(
                "Repeated settings remain eligible. The final result will "
                "report any overlap with previous data.",
                kind="warn",
            )
        )
    mo.vstack(
        [
            mo.md(f"**Previous data:** {stage2_context['previous_label']}"),
            mo.hstack(_stage2_notes, widths="equal"),
            *_note_blocks,
        ]
    )
    return


@app.cell
def _(mo, stage2_context):
    _stage2_previous_table = mo.ui.table(
        stage2_context["previous"],
        selection=None,
        pagination=True,
        page_size=12,
        show_column_summaries=True,
        show_download=True,
    )
    _stage2_previous_table
    return


@app.cell
def _(make_visualization_action_widgets):
    (
        stage2_previous_figure_format_widget,
        stage2_previous_pair_select_all_button,
        stage2_previous_pair_clear_all_button,
        stage2_previous_histogram_select_all_button,
        stage2_previous_histogram_clear_all_button,
    ) = make_visualization_action_widgets("Stage 2 previous-data")
    return (
        stage2_previous_figure_format_widget,
        stage2_previous_histogram_clear_all_button,
        stage2_previous_histogram_select_all_button,
        stage2_previous_pair_clear_all_button,
        stage2_previous_pair_select_all_button,
    )


@app.cell
def _(make_visualization_count_widgets, stage2_context):
    stage2_previous_input_columns = list(
        stage2_context["setup"].roles.inputs
    )
    (
        stage2_previous_available_pair_count,
        stage2_previous_pair_plot_limit,
        stage2_previous_histogram_limit,
        stage2_previous_pair_plot_count_widget,
        stage2_previous_histogram_count_widget,
    ) = make_visualization_count_widgets(
        stage2_previous_input_columns,
        "Stage 2 previous-data",
    )
    return (
        stage2_previous_available_pair_count,
        stage2_previous_histogram_count_widget,
        stage2_previous_histogram_limit,
        stage2_previous_input_columns,
        stage2_previous_pair_plot_count_widget,
        stage2_previous_pair_plot_limit,
    )


@app.cell
def _(
    make_visualization_selectors,
    stage2_previous_histogram_count_widget,
    stage2_previous_input_columns,
    stage2_previous_pair_plot_count_widget,
):
    (
        stage2_previous_pair_x_selectors,
        stage2_previous_pair_y_selectors,
        stage2_previous_histogram_selectors,
    ) = make_visualization_selectors(
        stage2_previous_input_columns,
        int(stage2_previous_pair_plot_count_widget.value or 0),
        int(stage2_previous_histogram_count_widget.value or 0),
    )
    return (
        stage2_previous_histogram_selectors,
        stage2_previous_pair_x_selectors,
        stage2_previous_pair_y_selectors,
    )


@app.cell
def _(
    make_visualization_save_selectors,
    stage2_previous_histogram_clear_all_button,
    stage2_previous_histogram_count_widget,
    stage2_previous_histogram_select_all_button,
    stage2_previous_pair_clear_all_button,
    stage2_previous_pair_plot_count_widget,
    stage2_previous_pair_select_all_button,
):
    stage2_previous_pair_save_selectors = (
        make_visualization_save_selectors(
            int(stage2_previous_pair_plot_count_widget.value or 0),
            stage2_previous_pair_select_all_button,
            stage2_previous_pair_clear_all_button,
        )
    )
    stage2_previous_histogram_save_selectors = (
        make_visualization_save_selectors(
            int(stage2_previous_histogram_count_widget.value or 0),
            stage2_previous_histogram_select_all_button,
            stage2_previous_histogram_clear_all_button,
        )
    )
    return (
        stage2_previous_histogram_save_selectors,
        stage2_previous_pair_save_selectors,
    )


@app.cell
def _(
    CANDIDATE_POINT_COLOR,
    PREVIOUS_POINT_COLOR,
    build_visualization_figures,
    histogram_bins_widget,
    stage2_context,
    stage2_previous_available_pair_count,
    stage2_previous_histogram_count_widget,
    stage2_previous_histogram_limit,
    stage2_previous_histogram_selectors,
    stage2_previous_input_columns,
    stage2_previous_pair_plot_count_widget,
    stage2_previous_pair_plot_limit,
    stage2_previous_pair_x_selectors,
    stage2_previous_pair_y_selectors,
):
    stage2_previous_visualization_figures = build_visualization_figures(
        heading="#### Previous Data Visualization",
        input_columns=stage2_previous_input_columns,
        available_pair_count=stage2_previous_available_pair_count,
        pair_limit=stage2_previous_pair_plot_limit,
        histogram_limit=stage2_previous_histogram_limit,
        pair_count_widget=stage2_previous_pair_plot_count_widget,
        histogram_count_widget=stage2_previous_histogram_count_widget,
        histogram_bins_widget=histogram_bins_widget,
        pair_x_selectors=stage2_previous_pair_x_selectors,
        pair_y_selectors=stage2_previous_pair_y_selectors,
        histogram_selectors=stage2_previous_histogram_selectors,
        scatter_series=[
            {
                "data": stage2_context["setup"].candidate,
                "name": "Candidate points",
                "color": CANDIDATE_POINT_COLOR,
                "size": 5,
            },
            {
                "data": stage2_context["previous"],
                "name": "Previous data",
                "color": PREVIOUS_POINT_COLOR,
                "size": 5,
            },
        ],
        histogram_series=[
            {
                "data": stage2_context["previous"],
                "name": "Previous data",
                "color": PREVIOUS_POINT_COLOR,
                "opacity": 0.50,
            }
        ],
        histogram_reference=stage2_context["setup"].candidate,
        show_legends=True,
    )
    return (stage2_previous_visualization_figures,)


@app.cell
def _(
    assemble_visualization_panel,
    stage2_previous_histogram_clear_all_button,
    stage2_previous_histogram_save_selectors,
    stage2_previous_histogram_select_all_button,
    stage2_previous_pair_clear_all_button,
    stage2_previous_pair_save_selectors,
    stage2_previous_pair_select_all_button,
    stage2_previous_visualization_figures,
):
    (
        _stage2_previous_visualization,
        stage2_previous_selected_scatter_plots,
        stage2_previous_selected_histogram_plots,
    ) = assemble_visualization_panel(
        stage2_previous_visualization_figures,
        pair_save_selectors=stage2_previous_pair_save_selectors,
        histogram_save_selectors=(
            stage2_previous_histogram_save_selectors
        ),
        pair_select_all_button=stage2_previous_pair_select_all_button,
        pair_clear_all_button=stage2_previous_pair_clear_all_button,
        histogram_select_all_button=(
            stage2_previous_histogram_select_all_button
        ),
        histogram_clear_all_button=(
            stage2_previous_histogram_clear_all_button
        ),
    )
    _stage2_previous_visualization
    return (
        stage2_previous_selected_histogram_plots,
        stage2_previous_selected_scatter_plots,
    )


@app.cell(hide_code=True)
def _(
    build_visualization_export_panel,
    stage2_previous_figure_format_widget,
    stage2_previous_selected_histogram_plots,
    stage2_previous_selected_scatter_plots,
):
    build_visualization_export_panel(
        figure_format_widget=stage2_previous_figure_format_widget,
        scatter_plots=stage2_previous_selected_scatter_plots,
        histogram_plots=stage2_previous_selected_histogram_plots,
        filename_prefix="stage2_previous_data",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configure and Run

    `Additional design size` is the number of new rows requested, not the
    total across stages. With the default settings this is **+6**, targeting
    **18 unique runs** after the selected 12-run first stage.
    """)
    return


@app.cell
def _(make_common_stage_widgets, mo, stage2_context, validate_stage2_settings):
    _stage2_candidate_count = len(stage2_context["setup"].candidate)
    _stage2_default_size = min(6, _stage2_candidate_count)
    _stage2_batch = mo.md(
        """
        | Setting | Value |
        |:--|:--|
        | Criterion | {criterion} |
        | Additional design size | {additional_size} |
        | Random restarts | {restarts} |
        | Calibration restarts | {calibration_restarts} |
        | Random seed | {random_seed} |
        """
    ).batch(
        additional_size=mo.ui.number(
            start=1,
            stop=_stage2_candidate_count,
            step=1,
            value=_stage2_default_size,
        ),
        **make_common_stage_widgets(2027),
    )
    stage2_settings_form = _stage2_batch.form(
        submit_button_label="Apply Stage 2 settings",
        validate=validate_stage2_settings,
        clear_on_submit=False,
    )
    stage2_settings_form
    return (stage2_settings_form,)


@app.cell
def _(mo, stage2_context, stage2_settings_form):
    mo.stop(
        stage2_settings_form.value is None,
        mo.callout(
            "Apply the Stage 2 settings to enable estimation and search.",
            kind="info",
        ),
    )
    _stage2_values = stage2_settings_form.value
    stage2_config = {
        "mode": str(_stage2_values["criterion"]),
        "additional_size": int(_stage2_values["additional_size"]),
        "num_restarts": int(_stage2_values["restarts"]),
        "calibration_restarts": int(
            _stage2_values["calibration_restarts"]
        ),
        "random_seed": int(_stage2_values["random_seed"]),
    }
    mo.stop(
        stage2_config["additional_size"]
        > len(stage2_context["setup"].candidate),
        mo.callout(
            "The additional design size exceeds the available candidate count.",
            kind="danger",
        ),
    )
    return (stage2_config,)


@app.cell
def _(mo, stage2_config, stage2_context):
    stage2_estimate_button = mo.ui.run_button(
        label="Estimate Stage 2 runtime",
        kind="neutral",
    )
    stage2_run_button = mo.ui.run_button(
        label="Run Stage 2",
        kind="success",
    )
    _target_total = (
        len(stage2_context["previous"])
        + stage2_config["additional_size"]
    )
    _stage2_restart_note = (
        mo.callout(
            "This setting can be expensive. Estimate runtime before running.",
            kind="warn",
        )
        if stage2_config["num_restarts"] >= 100_000
        else mo.md(
            f"Requested augmentation: **+{stage2_config['additional_size']}** "
            f"for a nominal total of **{_target_total}**."
        )
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"""
                    **Applied Stage 2 settings**

                    `{stage2_config['mode']}` ·
                    `+{stage2_config['additional_size']}` runs ·
                    `{stage2_config['num_restarts']:,}` restarts ·
                    `{stage2_config['calibration_restarts']:,}` calibration
                    restarts · seed `{stage2_config['random_seed']}`

                    The buttons below use this submitted snapshot. Apply the
                    form again after changing any visible setting.
                    """
                ),
                kind="info",
            ),
            _stage2_restart_note,
            mo.hstack(
                [stage2_estimate_button, stage2_run_button],
                justify="start",
            ),
        ]
    )
    return stage2_estimate_button, stage2_run_button


@app.cell
def _(
    estimate_uniform_runtime,
    format_duration,
    mo,
    stage2_config,
    stage2_context,
    stage2_estimate_button,
):
    mo.stop(not stage2_estimate_button.value)
    with mo.status.spinner(
        title="Calibrating Stage 2 runtime",
        subtitle=(
            f"{stage2_config['calibration_restarts']:,} calibration restarts"
        ),
    ):
        _stage2_estimate = estimate_uniform_runtime(
            setup=stage2_context["setup"],
            design_sizes=[stage2_config["additional_size"]],
            target_restarts=stage2_config["num_restarts"],
            mode=stage2_config["mode"],
            calibration_restarts=stage2_config["calibration_restarts"],
            random_state=stage2_config["random_seed"],
        )
    mo.callout(
        "Estimated Stage 2 runtime: "
        + format_duration(_stage2_estimate["total_seconds"]),
        kind="info",
    )
    return


@app.cell
def _(
    DEFAULT_HISTOGRAM_BINS,
    create_run_dirs,
    design_uniform_batch,
    mo,
    new_run_stamp,
    output_dir,
    pair_figure,
    pd,
    save_figure_artifact,
    stage2_config,
    stage2_context,
    stage2_run_button,
    time,
    write_csv,
    write_json,
    write_runtime_estimate,
):
    mo.stop(not stage2_run_button.value)
    _stage2_run_stamp = new_run_stamp()
    _stage2_dir, _stage2_design_dir, _stage2_plot_dir = create_run_dirs(
        output_dir, "stage2", stage2_config["mode"], _stage2_run_stamp
    )
    _stage2_warnings: list[str] = []
    _stage2_setup = stage2_context["setup"]
    _stage2_inputs = list(_stage2_setup.roles.inputs)

    with mo.status.spinner(
        title="Running Stage 2",
        subtitle=(
            f"+{stage2_config['additional_size']} design, "
            f"{stage2_config['num_restarts']:,} restarts"
        ),
        remove_on_exit=False,
    ):
        _previous_source_filename = (
            "stage1_selected_design.csv"
            if stage2_context["previous_source"] == "stage1"
            else "uploaded_previous_data.csv"
        )
        write_csv(
            stage2_context["previous"],
            output_dir / _previous_source_filename,
        )
        write_csv(
            stage2_context["previous"],
            output_dir / "stage2_previous_data.csv",
        )
        _stage2_setup_export_figure = pair_figure(
            stage2_context["previous"],
            _stage2_inputs,
            candidate=_stage2_setup.candidate,
            title="Stage 2 setup: previous data against candidate set",
            target_bins=DEFAULT_HISTOGRAM_BINS,
        )
        save_figure_artifact(
            _stage2_setup_export_figure,
            output_dir / "stage2_setup_pairwise.pdf",
            _stage2_warnings,
        )

        _stage2_estimate_payload = write_runtime_estimate(
            _stage2_setup,
            [stage2_config["additional_size"]],
            stage2_config,
            _stage2_dir,
        )

        _stage2_started = time.perf_counter()
        _stage2_result = design_uniform_batch(
            setup=_stage2_setup,
            design_sizes=[stage2_config["additional_size"]],
            num_restarts=stage2_config["num_restarts"],
            mode=stage2_config["mode"],
            random_state=stage2_config["random_seed"],
        )[0]
        _stage2_elapsed = time.perf_counter() - _stage2_started
        _combined_design = pd.concat(
            [stage2_context["previous"], _stage2_result.design],
            ignore_index=True,
        )
        _previous_keys = set(
            map(
                tuple,
                stage2_context["previous"][_stage2_inputs].itertuples(
                    index=False, name=None
                ),
            )
        )
        _additional_keys = list(
            map(
                tuple,
                _stage2_result.design[_stage2_inputs].itertuples(
                    index=False, name=None
                ),
            )
        )
        _overlap_count = sum(
            _key in _previous_keys for _key in _additional_keys
        )
        _unique_total = len(
            _combined_design.drop_duplicates(subset=_stage2_inputs)
        )
        _nominal_total = len(stage2_context["previous"]) + len(
            _stage2_result.design
        )
        if stage2_context["exclude_previous"] and _overlap_count:
            raise RuntimeError(
                "Repeated runs were selected even though exclusion was enabled."
            )

        _additional_path = (
            _stage2_design_dir
            / (
                f"stage2_{stage2_config['mode']}_additional_"
                f"{_stage2_result.design_size:02d}.csv"
            )
        )
        _combined_path = (
            _stage2_design_dir
            / (
                f"stage2_{stage2_config['mode']}_combined_total_"
                f"{_nominal_total:02d}.csv"
            )
        )
        _stage2_plot_path = (
            _stage2_plot_dir
            / (
                f"stage2_{stage2_config['mode']}_additional_"
                f"{_stage2_result.design_size:02d}_pairwise.pdf"
            )
        )
        write_csv(_stage2_result.design, _additional_path)
        write_csv(_combined_design, _combined_path)

        _stage2_figure = pair_figure(
            _stage2_result.design,
            _stage2_inputs,
            candidate=_stage2_setup.candidate,
            previous=stage2_context["previous"],
            title=(
                f"Stage 2 {stage2_config['mode']} augmentation, "
                f"+{_stage2_result.design_size} "
                f"(unique total {_unique_total}), "
                f"criterion={_stage2_result.criterion_value:.6f}"
            ),
            target_bins=DEFAULT_HISTOGRAM_BINS,
        )
        _saved_stage2_plot_relative = save_figure_artifact(
            _stage2_figure,
            _stage2_plot_path,
            _stage2_warnings,
            relative_to=output_dir,
        )

        _stage2_summary = pd.DataFrame(
            [
                {
                    "mode": _stage2_result.mode,
                    "additional_design_size": _stage2_result.design_size,
                    "previous_rows": len(stage2_context["previous"]),
                    "nominal_total_size": _nominal_total,
                    "unique_total_size": _unique_total,
                    "overlap_with_previous": _overlap_count,
                    "num_restarts": _stage2_result.num_restarts,
                    "criterion_value": _stage2_result.criterion_value,
                    "elapsed_seconds": _stage2_result.elapsed_time,
                    "additional_design_file": str(
                        _additional_path.relative_to(output_dir)
                    ),
                    "combined_design_file": str(
                        _combined_path.relative_to(output_dir)
                    ),
                    "pair_plot_artifact": _saved_stage2_plot_relative,
                }
            ]
        )
        write_csv(
            _stage2_summary,
            _stage2_dir / "created_designs_summary.csv",
        )
        write_csv(
            _stage2_summary,
            output_dir / "stage2_created_designs_summary.csv",
        )
        write_json(
            {
                **stage2_config,
                "previous_rows": len(stage2_context["previous"]),
                "nominal_total_size": _nominal_total,
                "unique_total_size": _unique_total,
                "overlap_with_previous": _overlap_count,
                "exclude_previous_candidates": stage2_context[
                    "exclude_previous"
                ],
                "elapsed_seconds": _stage2_elapsed,
                "bounds": _stage2_setup.bounds,
                "input_columns": _stage2_inputs,
                "index_column": _stage2_setup.roles.index,
            },
            _stage2_dir / "run_summary.json",
        )

    stage2_run = {
        "config": dict(stage2_config),
        "context": stage2_context,
        "estimate": _stage2_estimate_payload,
        "result": _stage2_result,
        "additional": _stage2_result.design,
        "combined": _combined_design,
        "summary": _stage2_summary,
        "nominal_total": _nominal_total,
        "unique_total": _unique_total,
        "overlap_count": _overlap_count,
        "elapsed_seconds": _stage2_elapsed,
        "directory": _stage2_dir,
        "warnings": _stage2_warnings,
    }
    _stage2_kind = (
        "warn"
        if _overlap_count or _stage2_warnings
        else "success"
    )
    mo.callout(
        (
            f"Created {_stage2_result.design_size} requested rows in "
            f"{_stage2_elapsed:.2f} seconds: {_unique_total} unique runs "
            f"across both stages, with {_overlap_count} repeated setting(s). "
            f"Results were saved under {_stage2_dir}."
        )
        + (
            "\n\nStatic export warnings:\n- "
            + "\n- ".join(_stage2_warnings)
            if _stage2_warnings
            else ""
        ),
        kind=_stage2_kind,
    )
    return (stage2_run,)


@app.cell
def _(csv_bytes, mo, output_dir, stage2_run, zip_directory):
    _stage2_summary_table = mo.ui.table(
        stage2_run["summary"],
        selection=None,
        pagination=False,
        show_column_summaries=True,
        show_download=True,
    )
    _additional_table = mo.ui.table(
        stage2_run["additional"],
        selection=None,
        pagination=True,
        page_size=10,
        show_column_summaries=True,
        show_download=True,
    )
    _combined_table = mo.ui.table(
        stage2_run["combined"],
        selection=None,
        pagination=True,
        page_size=18,
        show_column_summaries=True,
        show_download=True,
    )
    _additional_download = mo.download(
        data=lambda: csv_bytes(stage2_run["additional"]),
        filename=(
            f"stage2_{stage2_run['config']['mode']}_"
            f"additional_{len(stage2_run['additional']):02d}.csv"
        ),
        mimetype="text/csv",
        label="Download additional runs",
    )
    _combined_download = mo.download(
        data=lambda: csv_bytes(stage2_run["combined"]),
        filename=(
            f"stage2_{stage2_run['config']['mode']}_"
            f"combined_total_{stage2_run['nominal_total']:02d}.csv"
        ),
        mimetype="text/csv",
        label="Download combined design",
    )
    _artifact_download = mo.download(
        data=lambda: zip_directory(output_dir),
        filename=f"{output_dir.name}.zip",
        mimetype="application/zip",
        label="Download all CSV, JSON, and plot artifacts",
    )
    _overlap_callout = (
        mo.callout(
            f"{stage2_run['overlap_count']} Stage 2 row(s) repeat a previous "
            "input setting. Use the unique total when planning experiments.",
            kind="warn",
        )
        if stage2_run["overlap_count"]
        else mo.callout(
            "No Stage 2 input settings repeat previous data.",
            kind="success",
        )
    )
    mo.vstack(
        [
            mo.md("### Inspect Stage 2 Designs"),
            mo.hstack(
                [
                    mo.stat(
                        len(stage2_run["context"]["previous"]),
                        label="Previous runs",
                        bordered=True,
                    ),
                    mo.stat(
                        f"+{len(stage2_run['additional'])}",
                        label="Requested new runs",
                        bordered=True,
                    ),
                    mo.stat(
                        stage2_run["unique_total"],
                        label="Unique total",
                        bordered=True,
                    ),
                    mo.stat(
                        stage2_run["overlap_count"],
                        label="Repeated settings",
                        bordered=True,
                    ),
                ],
                widths="equal",
                wrap=True,
            ),
            _overlap_callout,
            _stage2_summary_table,
            mo.ui.tabs(
                {
                    "Additional rows": _additional_table,
                    "Combined design": _combined_table,
                },
                lazy=True,
            ),
            mo.hstack(
                [
                    _additional_download,
                    _combined_download,
                    _artifact_download,
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(make_visualization_action_widgets):
    (
        stage2_design_figure_format_widget,
        stage2_design_pair_select_all_button,
        stage2_design_pair_clear_all_button,
        stage2_design_histogram_select_all_button,
        stage2_design_histogram_clear_all_button,
    ) = make_visualization_action_widgets("Stage 2 design")
    return (
        stage2_design_figure_format_widget,
        stage2_design_histogram_clear_all_button,
        stage2_design_histogram_select_all_button,
        stage2_design_pair_clear_all_button,
        stage2_design_pair_select_all_button,
    )


@app.cell
def _(make_visualization_count_widgets, stage2_run):
    stage2_design_input_columns = list(
        stage2_run["context"]["setup"].roles.inputs
    )
    (
        stage2_design_available_pair_count,
        stage2_design_pair_plot_limit,
        stage2_design_histogram_limit,
        stage2_design_pair_plot_count_widget,
        stage2_design_histogram_count_widget,
    ) = make_visualization_count_widgets(
        stage2_design_input_columns,
        "Stage 2 design",
    )
    return (
        stage2_design_available_pair_count,
        stage2_design_histogram_count_widget,
        stage2_design_histogram_limit,
        stage2_design_input_columns,
        stage2_design_pair_plot_count_widget,
        stage2_design_pair_plot_limit,
    )


@app.cell
def _(
    make_visualization_selectors,
    stage2_design_histogram_count_widget,
    stage2_design_input_columns,
    stage2_design_pair_plot_count_widget,
):
    (
        stage2_design_pair_x_selectors,
        stage2_design_pair_y_selectors,
        stage2_design_histogram_selectors,
    ) = make_visualization_selectors(
        stage2_design_input_columns,
        int(stage2_design_pair_plot_count_widget.value or 0),
        int(stage2_design_histogram_count_widget.value or 0),
    )
    return (
        stage2_design_histogram_selectors,
        stage2_design_pair_x_selectors,
        stage2_design_pair_y_selectors,
    )


@app.cell
def _(
    make_visualization_save_selectors,
    stage2_design_histogram_clear_all_button,
    stage2_design_histogram_count_widget,
    stage2_design_histogram_select_all_button,
    stage2_design_pair_clear_all_button,
    stage2_design_pair_plot_count_widget,
    stage2_design_pair_select_all_button,
):
    stage2_design_pair_save_selectors = (
        make_visualization_save_selectors(
            int(stage2_design_pair_plot_count_widget.value or 0),
            stage2_design_pair_select_all_button,
            stage2_design_pair_clear_all_button,
        )
    )
    stage2_design_histogram_save_selectors = (
        make_visualization_save_selectors(
            int(stage2_design_histogram_count_widget.value or 0),
            stage2_design_histogram_select_all_button,
            stage2_design_histogram_clear_all_button,
        )
    )
    return (
        stage2_design_histogram_save_selectors,
        stage2_design_pair_save_selectors,
    )


@app.cell
def _(
    CANDIDATE_POINT_COLOR,
    DESIGN_POINT_COLOR,
    PREVIOUS_POINT_COLOR,
    build_visualization_figures,
    histogram_bins_widget,
    stage2_design_available_pair_count,
    stage2_design_histogram_count_widget,
    stage2_design_histogram_limit,
    stage2_design_histogram_selectors,
    stage2_design_input_columns,
    stage2_design_pair_plot_count_widget,
    stage2_design_pair_plot_limit,
    stage2_design_pair_x_selectors,
    stage2_design_pair_y_selectors,
    stage2_run,
):
    stage2_design_visualization_figures = build_visualization_figures(
        heading="#### Stage 2 Design Visualization",
        input_columns=stage2_design_input_columns,
        available_pair_count=stage2_design_available_pair_count,
        pair_limit=stage2_design_pair_plot_limit,
        histogram_limit=stage2_design_histogram_limit,
        pair_count_widget=stage2_design_pair_plot_count_widget,
        histogram_count_widget=stage2_design_histogram_count_widget,
        histogram_bins_widget=histogram_bins_widget,
        pair_x_selectors=stage2_design_pair_x_selectors,
        pair_y_selectors=stage2_design_pair_y_selectors,
        histogram_selectors=stage2_design_histogram_selectors,
        scatter_series=[
            {
                "data": stage2_run["context"]["setup"].candidate,
                "name": "Candidate points",
                "color": CANDIDATE_POINT_COLOR,
                "size": 5,
            },
            {
                "data": stage2_run["context"]["previous"],
                "name": "Previous data",
                "color": PREVIOUS_POINT_COLOR,
                "size": 5,
            },
            {
                "data": stage2_run["additional"],
                "name": "Design points",
                "color": DESIGN_POINT_COLOR,
                "size": 6,
            },
        ],
        histogram_series=[
            {
                "data": stage2_run["additional"],
                "name": "Design points",
                "color": DESIGN_POINT_COLOR,
                "opacity": 0.75,
            },
            {
                "data": stage2_run["context"]["previous"],
                "name": "Previous data",
                "color": PREVIOUS_POINT_COLOR,
                "opacity": 0.50,
            },
        ],
        histogram_reference=stage2_run["context"]["setup"].candidate,
        show_legends=True,
    )
    return (stage2_design_visualization_figures,)


@app.cell
def _(
    assemble_visualization_panel,
    stage2_design_histogram_clear_all_button,
    stage2_design_histogram_save_selectors,
    stage2_design_histogram_select_all_button,
    stage2_design_pair_clear_all_button,
    stage2_design_pair_save_selectors,
    stage2_design_pair_select_all_button,
    stage2_design_visualization_figures,
):
    (
        _stage2_design_visualization,
        stage2_design_selected_scatter_plots,
        stage2_design_selected_histogram_plots,
    ) = assemble_visualization_panel(
        stage2_design_visualization_figures,
        pair_save_selectors=stage2_design_pair_save_selectors,
        histogram_save_selectors=stage2_design_histogram_save_selectors,
        pair_select_all_button=stage2_design_pair_select_all_button,
        pair_clear_all_button=stage2_design_pair_clear_all_button,
        histogram_select_all_button=(
            stage2_design_histogram_select_all_button
        ),
        histogram_clear_all_button=(
            stage2_design_histogram_clear_all_button
        ),
    )
    _stage2_design_visualization
    return (
        stage2_design_selected_histogram_plots,
        stage2_design_selected_scatter_plots,
    )


@app.cell(hide_code=True)
def _(
    build_visualization_export_panel,
    stage2_design_figure_format_widget,
    stage2_design_selected_histogram_plots,
    stage2_design_selected_scatter_plots,
    stage2_run,
):
    build_visualization_export_panel(
        figure_format_widget=stage2_design_figure_format_widget,
        scatter_plots=stage2_design_selected_scatter_plots,
        histogram_plots=stage2_design_selected_histogram_plots,
        filename_prefix=(
            f"stage2_{stage2_run['config']['mode']}_additional_"
            f"{len(stage2_run['additional']):02d}"
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Notes for interpretation

    - More random restarts increase the opportunity to find a better
      design, but cost more time. Use the estimator before large searches.
    - Bounds normalize the inputs used in distance calculations. Compare
      criterion values only between designs using the same inputs, bounds,
      criterion, and previous data.
    - The candidate rows define the allowed sampling region. Non-input
      columns are excluded from optimization but retained in selected rows
      for identification and traceability.

    Launch from the repository root with:

    ```bash
    marimo edit examples/example-uniform-5d-marimo.py
    ```

    Use `marimo run` instead of `marimo edit` for a read-only app.
    """)
    return


if __name__ == "__main__":
    app.run()
