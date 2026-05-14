"""Basic time features and cyclical encodings.

These are cheap, deterministic, and known to help radiation models. Mirrors
the starter notebook's set with a slightly cleaner API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CYCLIC_PERIODS = {
    "hour": 24,
    "month": 12,
    "day_of_year": 365.25,
}


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    include_raw: list[str] | None = None,
    cyclical: list[str] | None = None,
) -> pd.DataFrame:
    """Add calendar + cyclical time features. Does not mutate input."""
    include_raw = include_raw or [
        "year", "month", "day", "hour", "minute", "day_of_week", "day_of_year"
    ]
    cyclical = cyclical or ["hour", "month", "day_of_year"]

    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")

    raw_map = {
        "year": ts.dt.year,
        "month": ts.dt.month,
        "day": ts.dt.day,
        "hour": ts.dt.hour,
        "minute": ts.dt.minute,
        "day_of_week": ts.dt.dayofweek,
        "day_of_year": ts.dt.dayofyear,
        "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
    }
    for name in include_raw:
        if name in raw_map:
            out[name] = raw_map[name]
    if "is_weekend" in include_raw:
        out["is_weekend"] = raw_map["is_weekend"]

    for name in cyclical:
        if name not in _CYCLIC_PERIODS:
            raise ValueError(f"unknown cyclical feature: {name}")
        period = _CYCLIC_PERIODS[name]
        values = raw_map[name]
        out[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    return out
