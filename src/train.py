"""Train the final model on all training data and persist it.

Reads the best_iteration from a prior CV experiment (mean across folds) and
trains a single LightGBM on the full training set for that many rounds.
Saves the model + metadata so predict.py can load it without re-running CV.

Usage:
    python -m src.train --config config/config.yaml \
        --features config/features.yaml --model config/model.yaml \
        --experiment exp_001_lgbm_baseline
"""
from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from src.features.build_features import build_feature_matrix
from src.models.lgbm import LightGBMRegressor
from src.utils.io import load_config, read_parquet, resolve_path
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def _read_cv_best_iter(exp_dir, default: int) -> int:
    """Mean best_iteration across CV folds, rounded up. Falls back to default."""
    path = exp_dir / "fold_scores.csv"
    if not path.exists():
        return default
    df = pd.read_csv(path)
    if "best_iteration" not in df.columns or df["best_iteration"].dropna().empty:
        return default
    return int(math.ceil(df["best_iteration"].dropna().mean()))


def main(config_path: str, features_path: str, model_path: str, experiment: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)
    mcfg = load_config(model_path)
    set_seed(cfg["project"]["seed"])

    active = mcfg["active"]
    if active != "lightgbm":
        raise NotImplementedError(f"Only lightgbm is wired up so far (got '{active}').")

    exp_dir = resolve_path(cfg["paths"]["experiments_dir"]) / experiment
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading interim train parquet...")
    train = read_parquet(cfg["paths"]["train_interim"])
    target_col = cfg["columns"]["target"]
    y = train[target_col].to_numpy()

    log.info("Building feature matrix...")
    X, cat_features, station_categories = build_feature_matrix(train, cfg, fcfg)
    log.info("X shape: %s, categorical: %s", X.shape, cat_features)

    default_rounds = mcfg["lightgbm"]["training"]["num_boost_round"]
    best_iter = _read_cv_best_iter(exp_dir, default_rounds)
    log.info("Training rounds (mean of CV best_iter, fallback default): %d", best_iter)

    training = dict(mcfg["lightgbm"]["training"])
    training["num_boost_round"] = best_iter
    training["early_stopping_rounds"] = 0   # no holdout in final fit

    model = LightGBMRegressor(mcfg["lightgbm"]["params"], training)
    model.fit(X, y, categorical_features=cat_features)

    model_path_out = models_dir / f"{experiment}.txt"
    model.booster_.save_model(str(model_path_out))

    metadata = {
        "experiment": experiment,
        "model_type": active,
        "feature_names": list(X.columns),
        "categorical_features": cat_features,
        "station_categories": station_categories,
        "num_boost_round": best_iter,
        "n_train_rows": int(len(y)),
    }
    meta_path = models_dir / f"{experiment}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved model: %s", model_path_out)
    log.info("Saved metadata: %s", meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    parser.add_argument("--model", default="config/model.yaml")
    parser.add_argument("--experiment", default="exp_001_lgbm_baseline")
    args = parser.parse_args()
    main(args.config, args.features, args.model, args.experiment)