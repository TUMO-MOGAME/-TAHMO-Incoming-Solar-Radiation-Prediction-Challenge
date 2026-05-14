"""Cross-validation splitters.

Three strategies are useful for this problem; the rest would mislead us.

  Primary:  LeaveOneMonthOutSplitter
            6 folds, one per odd month in train. Mimics the test task
            (predict an unseen month). Decisions are made on this.

  Sanity 1: GroupKFoldByYearSplitter
            One fold per calendar year (2016-2020). Catches over-fit to
            year 2018 (which is 68% of training rows).

  Sanity 2: TimeForwardSplitter
            Per-station forward-walking time splits. If this scores
            better than the primary, a feature is leaking future info.

Strategies we deliberately do not implement (would mislead on this data):
  - Random KFold: leaks across 15-min autocorrelation, produces
    optimistic scores that don't reflect test performance.
  - StratifiedKFold: classification tool; not appropriate for
    regression with a temporal generalization challenge.
  - GroupKFold-by-station: all 40 stations appear in both train and
    test, so holding out stations tests the wrong thing.
  - StratifiedGroupKFold: no useful axis to stratify on here.
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
    """Hold out one calendar month at a time as validation.

    Default `months=[1, 3, 5, 7, 9, 11]` matches the months actually
    present in the competition train set. Every station that has data
    in the held-out month contributes to valid, mirroring the
    competition where every station contributes to test.
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


class GroupKFoldByYearSplitter:
    """Hold out one calendar year at a time as validation.

    Folds are determined by the unique years present in `df` (sorted).
    With train spanning 2016-2020 this typically yields 5 folds. The
    2018 fold is the most informative because 2018 holds ~68% of train.
    """

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        self.timestamp_col = timestamp_col

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        if self.timestamp_col not in df.columns:
            raise KeyError(f"timestamp column '{self.timestamp_col}' missing")
        years = pd.to_datetime(df[self.timestamp_col]).dt.year.to_numpy()
        all_idx = np.arange(len(df))
        for fold_id, year in enumerate(sorted(np.unique(years))):
            valid_mask = years == year
            yield Fold(
                fold_id=fold_id,
                train_idx=all_idx[~valid_mask],
                valid_idx=all_idx[valid_mask],
                description=f"valid_year={int(year)}",
            )


class TimeForwardSplitter:
    """Per-station forward-walking time splits — leakage sanity check.

    Splits each station's timeline into (n_folds + 1) sequential chunks
    and uses each non-first chunk as a validation fold, with all prior
    chunks as training. The same global fold_id groups validations across
    stations so metrics aggregate naturally.
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


SPLITTER_REGISTRY = {
    "leave_one_month_out": "leave_one_month_out",
    "group_kfold_by_year": "group_kfold_by_year",
    "time_forward": "time_forward",
}


def _build_splitter(name: str, cfg: dict):
    ts_col = cfg["columns"]["timestamp"]
    station_col = cfg["columns"]["station"]
    cv = cfg["cv"]
    if name == "leave_one_month_out":
        return LeaveOneMonthOutSplitter(months=cv.get("train_months"), timestamp_col=ts_col)
    if name == "group_kfold_by_year":
        return GroupKFoldByYearSplitter(timestamp_col=ts_col)
    if name == "time_forward":
        return TimeForwardSplitter(
            n_folds=cv.get("n_folds_time_forward", 4),
            station_col=station_col,
            timestamp_col=ts_col,
        )
    raise ValueError(f"unknown splitter: {name}")


def make_splitters(cfg: dict) -> list[tuple[str, object]]:
    """Build the list of (name, splitter) configured in cfg.cv.strategies.

    Order matters — the first entry is the PRIMARY strategy used for
    decisions and downstream OOF predictions. Later entries are sanity
    checks reported in the dashboard.
    """
    names = cfg["cv"].get("strategies")
    if not names:
        raise ValueError("config.cv.strategies must be a non-empty list")
    return [(n, _build_splitter(n, cfg)) for n in names]