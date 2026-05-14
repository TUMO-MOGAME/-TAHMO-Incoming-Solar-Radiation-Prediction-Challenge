"""Cross-validation splitters.

The competition holds out *even months of Year 1* per station. Our primary
splitter mimics this by holding out even months from training. Random K-fold
over rows would leak across the 15-minute autocorrelation horizon and is
explicitly avoided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    description: str


class EvenMonthHoldoutSplitter:
    """Rotate held-out even months across folds.

    Example with n_folds=3 over even_months=[2, 4, 6, 8, 10, 12]:
        fold 0 -> validate on months {2, 4},  train on rest
        fold 1 -> validate on months {6, 8},  train on rest
        fold 2 -> validate on months {10, 12}, train on rest

    Splits are applied globally — every station contributes to both train and
    valid in each fold, matching how the competition holds out months per
    station rather than holding out whole stations.
    """

    def __init__(
        self,
        even_months: list[int] | None = None,
        n_folds: int = 3,
        timestamp_col: str = "timestamp",
    ) -> None:
        self.even_months = even_months or [2, 4, 6, 8, 10, 12]
        if any(m % 2 != 0 for m in self.even_months):
            raise ValueError("even_months must contain only even integers")
        self.n_folds = n_folds
        self.timestamp_col = timestamp_col

    def _month_chunks(self) -> list[list[int]]:
        """Partition even_months into n_folds contiguous chunks."""
        return [list(c) for c in np.array_split(np.array(self.even_months), self.n_folds)]

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        if self.timestamp_col not in df.columns:
            raise KeyError(f"timestamp column '{self.timestamp_col}' missing")
        months = pd.to_datetime(df[self.timestamp_col]).dt.month.to_numpy()
        all_idx = np.arange(len(df))
        chunks = self._month_chunks()
        for fold_id, valid_months in enumerate(chunks):
            valid_mask = np.isin(months, valid_months)
            yield Fold(
                fold_id=fold_id,
                train_idx=all_idx[~valid_mask],
                valid_idx=all_idx[valid_mask],
                description=f"valid_months={valid_months}",
            )


class TimeForwardSplitter:
    """Per-station forward-walking time splits — leakage sanity check.

    Splits each station's timeline into (n_folds + 1) sequential chunks and
    uses each non-first chunk as a validation fold, with all prior chunks as
    training. The same global fold_id groups validations across stations so
    metrics aggregate naturally.
    """

    def __init__(
        self,
        n_folds: int = 4,
        station_col: str = "station",
        timestamp_col: str = "timestamp",
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        self.n_folds = n_folds
        self.station_col = station_col
        self.timestamp_col = timestamp_col

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        if self.station_col not in df.columns:
            raise KeyError(f"station column '{self.station_col}' missing")
        if self.timestamp_col not in df.columns:
            raise KeyError(f"timestamp column '{self.timestamp_col}' missing")

        order = df.sort_values([self.station_col, self.timestamp_col]).index.to_numpy()
        df_sorted = df.loc[order]
        station_ids = df_sorted[self.station_col].to_numpy()

        per_station_chunk = np.empty(len(df_sorted), dtype=np.int64)
        for station, idxs in (
            pd.Series(np.arange(len(df_sorted)))
            .groupby(station_ids, sort=False)
            .groups.items()
        ):
            arr = np.asarray(idxs)
            chunks = np.array_split(arr, self.n_folds + 1)
            for chunk_id, chunk in enumerate(chunks):
                per_station_chunk[chunk] = chunk_id

        for fold_id in range(1, self.n_folds + 1):
            train_mask_sorted = per_station_chunk < fold_id
            valid_mask_sorted = per_station_chunk == fold_id
            train_idx = order[train_mask_sorted]
            valid_idx = order[valid_mask_sorted]
            yield Fold(
                fold_id=fold_id - 1,
                train_idx=np.sort(train_idx),
                valid_idx=np.sort(valid_idx),
                description=f"forward_chunk={fold_id}",
            )


def make_splitter(cfg: dict) -> EvenMonthHoldoutSplitter | TimeForwardSplitter:
    """Factory: build a splitter from the cv section of config.yaml."""
    cv = cfg["cv"]
    ts_col = cfg["columns"]["timestamp"]
    station_col = cfg["columns"]["station"]
    name = cv["splitter"]
    if name == "even_month_holdout":
        return EvenMonthHoldoutSplitter(
            even_months=cv.get("even_months"),
            n_folds=cv["n_folds"],
            timestamp_col=ts_col,
        )
    if name == "time_forward":
        return TimeForwardSplitter(
            n_folds=cv["n_folds"],
            station_col=station_col,
            timestamp_col=ts_col,
        )
    raise ValueError(f"unknown splitter: {name}")
