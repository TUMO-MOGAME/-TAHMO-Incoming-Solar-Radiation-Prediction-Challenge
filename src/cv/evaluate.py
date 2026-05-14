"""Evaluation metrics: MBE, RMSE, and the combined CV score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias Error. Positive = overprediction."""
    return float(np.mean(y_pred - y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def combined_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mbe_weight: float = 0.5,
    rmse_weight: float = 0.5,
) -> float:
    """Weighted combination of |MBE| and RMSE. Lower is better."""
    return mbe_weight * abs(mbe(y_true, y_pred)) + rmse_weight * rmse(y_true, y_pred)


@dataclass
class ScoreReport:
    n: int
    mbe: float
    abs_mbe: float
    rmse: float
    combined: float
    per_station: pd.DataFrame | None = None


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stations: np.ndarray | None = None,
    mbe_weight: float = 0.5,
    rmse_weight: float = 0.5,
) -> ScoreReport:
    """Compute overall and per-station scores."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = mbe(y_true, y_pred)
    r = rmse(y_true, y_pred)
    c = mbe_weight * abs(m) + rmse_weight * r

    per_station = None
    if stations is not None:
        stations = np.asarray(stations)
        df = pd.DataFrame({"station": stations, "y_true": y_true, "y_pred": y_pred})
        per_station = (
            df.groupby("station")
            .apply(
                lambda g: pd.Series({
                    "n": len(g),
                    "mbe": mbe(g["y_true"].values, g["y_pred"].values),
                    "rmse": rmse(g["y_true"].values, g["y_pred"].values),
                    "combined": combined_score(
                        g["y_true"].values, g["y_pred"].values,
                        mbe_weight, rmse_weight,
                    ),
                }),
                include_groups=False,
            )
            .reset_index()
        )

    return ScoreReport(
        n=len(y_true),
        mbe=m,
        abs_mbe=abs(m),
        rmse=r,
        combined=c,
        per_station=per_station,
    )
