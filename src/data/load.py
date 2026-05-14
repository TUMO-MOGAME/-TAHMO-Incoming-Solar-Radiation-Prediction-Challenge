"""Read raw competition CSVs into typed, sorted DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import resolve_path

_TIMESTAMP_COL = "timestamp"
_STATION_COL = "station"


def _read_and_type(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path))
    if _TIMESTAMP_COL in df.columns:
        df[_TIMESTAMP_COL] = pd.to_datetime(df[_TIMESTAMP_COL], errors="coerce")
    if _STATION_COL in df.columns:
        df = df.sort_values([_STATION_COL, _TIMESTAMP_COL]).reset_index(drop=True)
    return df


def read_train(path: str | Path) -> pd.DataFrame:
    """Read Train.csv: parsed timestamp, sorted by (station, timestamp)."""
    return _read_and_type(path)


def read_test(path: str | Path) -> pd.DataFrame:
    """Read Test.csv: parsed timestamp, sorted by (station, timestamp)."""
    return _read_and_type(path)


def read_sample_submission(path: str | Path) -> pd.DataFrame:
    """Read SampleSubmission.csv as-is."""
    return pd.read_csv(resolve_path(path))
