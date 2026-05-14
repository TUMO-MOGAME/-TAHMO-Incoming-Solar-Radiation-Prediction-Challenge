"""Station-level features and CV-safe target encoding.

Planned:
    - station as a categorical feature (no encoding needed for LightGBM/CatBoost)
    - CV-safe target mean encoding (fit per fold's train, applied to valid)
    - country / elevation bucket interactions
"""
from __future__ import annotations

import pandas as pd


def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: station-level features. Initially a passthrough."""
    return df


def fit_target_encoding(train_df: pd.DataFrame, target_col: str, group_col: str = "station") -> dict:
    """TODO: compute station-level target means on the training fold."""
    raise NotImplementedError("Target encoding not yet implemented.")


def apply_target_encoding(df: pd.DataFrame, encoding: dict, group_col: str = "station") -> pd.DataFrame:
    """TODO: apply the encoding produced by fit_target_encoding."""
    raise NotImplementedError("Target encoding not yet implemented.")
