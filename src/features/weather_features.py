"""Lag and rolling features for weather signals.

Planned: per-station lagged values and rolling means/stds of humidity,
temperature, precipitation at multiple windows. Must be computed within
station groups to avoid cross-station bleed.
"""
from __future__ import annotations

import pandas as pd


def add_weather_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
    station_col: str = "station",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """TODO: per-station lag + rolling features for each named column."""
    raise NotImplementedError(
        "Weather lag/rolling features not yet implemented."
    )
