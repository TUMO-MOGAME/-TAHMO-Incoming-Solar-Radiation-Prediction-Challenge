"""Generate predictions on the test set using a model saved by train.py.

Usage:
    python -m src.predict --config config/config.yaml \
        --features config/features.yaml --experiment exp_001_lgbm_baseline
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build_features import build_feature_matrix
from src.utils.io import load_config, read_parquet, resolve_path, write_parquet
from src.utils.logging import setup_logger


def main(config_path: str, features_path: str, experiment: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    model_file = models_dir / f"{experiment}.txt"
    meta_file = models_dir / f"{experiment}_metadata.json"
    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Model or metadata missing for '{experiment}'. Run train.py first."
        )

    with open(meta_file) as f:
        metadata = json.load(f)
    log.info("Loaded metadata: features=%d, categorical=%s, rounds=%d",
             len(metadata["feature_names"]), metadata["categorical_features"],
             metadata["num_boost_round"])

    booster = lgb.Booster(model_file=str(model_file))

    log.info("Loading interim test parquet...")
    test = read_parquet(cfg["paths"]["test_interim"])

    log.info("Building feature matrix with train station categories...")
    X_test, cat_features, _ = build_feature_matrix(
        test, cfg, fcfg,
        station_categories=metadata["station_categories"],
        source_path=cfg["paths"]["test_interim"],
    )
    X_test = X_test[metadata["feature_names"]]
    if cat_features != metadata["categorical_features"]:
        raise RuntimeError(
            f"categorical mismatch: train={metadata['categorical_features']} vs test={cat_features}"
        )

    log.info("Predicting %d rows...", len(X_test))
    raw_preds = booster.predict(X_test)
    clip_min = cfg["evaluation"]["clip_min"]
    clip_max = cfg["evaluation"]["clip_max"]
    preds = np.clip(raw_preds, clip_min, clip_max)
    log.info("Predictions: min=%.2f max=%.2f mean=%.2f", preds.min(), preds.max(), preds.mean())

    id_col = cfg["columns"]["id"]
    pred_df = pd.DataFrame({id_col: test[id_col].values, "prediction": preds})

    out_dir = resolve_path(cfg["paths"]["experiments_dir"]) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions.parquet"
    write_parquet(pred_df, str(out_path))
    log.info("Wrote test predictions: %s (rows=%d)", out_path, len(pred_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    parser.add_argument("--experiment", default="exp_001_lgbm_baseline")
    args = parser.parse_args()
    main(args.config, args.features, args.experiment)