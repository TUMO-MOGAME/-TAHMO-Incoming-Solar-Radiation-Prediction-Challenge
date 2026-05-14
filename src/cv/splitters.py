"""Cross-validation splitters.

EDA finding: train contains only odd months {1,3,5,7,9,11} and test contains
only even months {2,4,6,8,10,12}, across all 2016-2020. The competition is
"predict the months you have never seen." Random K-fold over rows would leak
across the 15-minute autocorrelation horizon and is explicitly avoided.

Primary CV: LeaveOneMonthOutSplitter holds out one odd month at a time —
the closest simulation of test conditions we can build from train alone.

Secondary CV: TimeForwardSplitter for leakage sanity checks.
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


class LeaveOneMonthOutSplitter:
    """Hold out one month at a time as validation.

    Default `months=[1, 3, 5, 7, 9, 11]` matches the months actually present
    in the competition train set. The model trained on 5 odd months is
    evaluated on the 6th — the closest analogue to test (even months) we
    can simulate from train alone.

    Splits are applied globally: every station that has data in the held-out
    month contributes to valid, mirroring the competition where every station
    contributes to test.
    """

    def __init__(
        self,
        months: list[int] | None = None,
        timestamp_col: str = "timestamp",
    ) -> None:
        self.months = months or [1, 3, 5, 7, 9, 11]
        self.timestamp_col = timestamp_col

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        if self.timestamp_col not in df.columns:
            raise KeyError(f"timestamp column '{self.timestamp_col}' missing")
        months = pd.to_datetime(df[self.timestamp_col]).dt.month.to_numpy()
        all_idx = np.arange(len(df))
        for fold_id, valid_month in enumerate(self.months):
            valid_mask = months == valid_month
            yield Fold(
                fold_id=fold_id,
                train_idx=all_idx[~valid_mask],
                valid_idx=all_idx[valid_mask],
                description=f"valid_month={valid_month}",
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


def make_splitter(cfg: dict) -> LeaveOneMonthOutSplitter | TimeForwardSplitter:
    """Factory: build a splitter from the cv section of config.yaml."""
    cv = cfg["cv"]
    ts_col = cfg["columns"]["timestamp"]
    station_col = cfg["columns"]["station"]
    name = cv["splitter"]
    if name == "leave_one_month_out":
        return LeaveOneMonthOutSplitter(
            months=cv.get("train_months"),
            timestamp_col=ts_col,
        )
    if name == "time_forward":
        return TimeForwardSplitter(
            n_folds=cv["n_folds"],
            station_col=station_col,
            timestamp_col=ts_col,
        )
    raise ValueError(f"unknown splitter: {name}")
