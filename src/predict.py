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
import xgboost as xgb

from src.features.build_features import build_feature_matrix
from src.utils.io import load_config, read_parquet, resolve_path, write_parquet
from src.utils.logging import setup_logger


def _load_booster(model_type: str, model_file):
    if model_type == "lightgbm":
        return lgb.Booster(model_file=str(model_file))
    if model_type == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(model_file))
        return booster
    raise NotImplementedError(f"Unknown model_type: {model_type}")


def _predict(booster, model_type: str, X: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    if model_type == "lightgbm":
        return booster.predict(X)
    if model_type == "xgboost":
        enable_cat = bool(cat_features)
        dmat = xgb.DMatrix(X, enable_categorical=enable_cat)
        return booster.predict(dmat)
    raise NotImplementedError(f"Unknown model_type: {model_type}")


def main(config_path: str, features_path: str, experiment: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    meta_file = models_dir / f"{experiment}_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata missing for '{experiment}'. Run train.py first.")

    with open(meta_file) as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type", "lightgbm")
    model_filename = metadata.get(
        "model_file",
        f"{experiment}.txt" if model_type == "lightgbm" else f"{experiment}.ubj",
    )
    model_file = models_dir / model_filename
    if not model_file.exists():
        raise FileNotFoundError(f"Model file missing: {model_file}")

    log.info("Model: %s | features=%d | categorical=%s | rounds=%d",
             model_type, len(metadata["feature_names"]),
             metadata["categorical_features"], metadata["num_boost_round"])

    booster = _load_booster(model_type, model_file)

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
    raw_preds = _predict(booster, model_type, X_test, cat_features)
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