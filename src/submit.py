"""Format predictions into the competition submission CSV.

Submission columns: ID, TargetMBE, TargetRMSE.

Default behaviour follows the baseline: write the same prediction to both
target columns. The TODO is to investigate whether a bias-corrected
prediction in TargetMBE yields a better leaderboard score.

Usage:
    python -m src.submit --config config/config.yaml --predictions <path.parquet> --out <path.csv>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.io import load_config, read_parquet, resolve_path
from src.utils.logging import setup_logger


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.DataFrame,
    id_col: str = "ID",
    pred_col: str = "prediction",
    clip_min: float = 0.0,
    clip_max: float = 1400.0,
) -> pd.DataFrame:
    """Merge predictions onto sample submission template, clip to physical range."""
    sub = sample_submission[[id_col, "TargetMBE", "TargetRMSE"]].copy()
    merged = sub.merge(predictions[[id_col, pred_col]], on=id_col, how="left")

    missing = merged[pred_col].isna().sum()
    if missing:
        raise ValueError(f"{missing} predictions missing after merge — check ID coverage")

    clipped = np.clip(merged[pred_col].to_numpy(), clip_min, clip_max)
    merged["TargetMBE"] = clipped
    merged["TargetRMSE"] = clipped
    return merged.drop(columns=[pred_col])


def main(config_path: str, predictions_path: str, out_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)

    log.info("Loading sample submission and predictions...")
    sample = read_parquet(cfg["paths"]["sample_submission_interim"])
    preds = read_parquet(predictions_path)

    submission = build_submission(
        sample, preds,
        id_col=cfg["columns"]["id"],
        clip_min=cfg["evaluation"]["clip_min"],
        clip_max=cfg["evaluation"]["clip_max"],
    )
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    log.info("Wrote submission: %s (rows=%d)", out, len(submission))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--predictions", required=True, help="parquet with columns ID, prediction")
    parser.add_argument("--out", required=True, help="path to write submission csv")
    args = parser.parse_args()
    main(args.config, args.predictions, args.out)
