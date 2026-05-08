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

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_csv(path: str | Path, *, index_column: str | None = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: CSV file path.
        index_column: Optional column to move into the DataFrame index.

    Returns:
        Loaded table with stripped column names.
    """
    frame = pd.read_csv(path)
    frame.rename(columns=lambda name: str(name).strip(), inplace=True)
    if index_column:
        frame.set_index(index_column, inplace=True)
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path, *, include_index: bool = False) -> None:
    """Write a DataFrame to CSV.

    Args:
        frame: Table to write.
        path: Output file path.
        include_index: If ``True``, include the DataFrame index in the file.
    """
    index_label = frame.index.name if include_index else None
    frame.to_csv(path, index=include_index, index_label=index_label)


def aggregate_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and aggregate one or more CSV tables.

    Args:
        paths: Iterable of CSV file paths.

    Returns:
        The single loaded table, or an inner-column concatenation of multiple
        tables with duplicate rows removed.
    """
    frames = [load_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one table is required.")
    if len(frames) == 1:
        return frames[0]
    merged = pd.concat(frames, join="inner", ignore_index=True)
    return merged.drop_duplicates().reset_index(drop=True)


def align_previous_to_candidate(candidate: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Reorder previous-data columns to match a candidate table.

    Args:
        candidate: Candidate table that defines column order.
        previous: Previous-data table with matching columns.

    Returns:
        Previous-data table reordered to the candidate column layout.
    """
    return previous[candidate.columns.tolist()]
