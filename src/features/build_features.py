"""Feature pipeline: turn an interim DataFrame into a model-ready feature matrix.

Shared by `src/cv/run_cv.py`, `src/train.py`, and `src/predict.py` so train
and test get the *identical* transformation. Also exposes a CLI that caches
processed parquets if you ever want them precomputed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.features.solar_geometry import SOLAR_FEATURE_COLS, add_solar_geometry_features
from src.features.time_features import add_time_features
from src.utils.io import load_config, read_parquet, resolve_path, write_parquet
from src.utils.logging import setup_logger

_BAD_CHARS = re.compile(r"[^0-9a-zA-Z_]")
_logger = setup_logger()


def sanitize_column_name(name: str) -> str:
    """LightGBM rejects JSON-special chars in feature names. Normalize once."""
    cleaned = _BAD_CHARS.sub("_", name)
    return re.sub(r"_+", "_", cleaned).strip("_")


def _solar_cache_path(cfg: dict, source_path: str) -> Path:
    """Deterministic cache path for solar features keyed by the source parquet."""
    src = resolve_path(source_path)
    stem = src.stem  # e.g. "train" or "test"
    return resolve_path(cfg["paths"]["processed_dir"]) / f"{stem}_solar.parquet"


def _attach_solar_features(
    df: pd.DataFrame,
    cfg: dict,
    source_path: str | None,
) -> pd.DataFrame:
    """Return df with solar feature columns attached. Cached to processed/ on disk.

    Cache key is keyed off the source parquet stem. If the source parquet is
    unknown (no `source_path`), we recompute every call.
    """
    if all(c in df.columns for c in SOLAR_FEATURE_COLS):
        _logger.info("Solar features already present, skipping recompute")
        return df

    if source_path is not None:
        cache_path = _solar_cache_path(cfg, source_path)
        if cache_path.exists():
            _logger.info("Loading cached solar features: %s", cache_path)
            cached = pd.read_parquet(cache_path)
            id_col = cfg["columns"]["id"]
            merged = df.merge(cached[[id_col, *SOLAR_FEATURE_COLS]], on=id_col, how="left")
            if merged[SOLAR_FEATURE_COLS].isna().any().any():
                _logger.warning("Cache had NaNs after merge — recomputing")
            else:
                return merged

    _logger.info("Computing solar features over %d rows...", len(df))
    out = add_solar_geometry_features(df, timestamp_col=cfg["columns"]["timestamp"])

    if source_path is not None:
        cache_path = _solar_cache_path(cfg, source_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        id_col = cfg["columns"]["id"]
        out[[id_col, *SOLAR_FEATURE_COLS]].to_parquet(cache_path, index=False)
        _logger.info("Cached solar features: %s", cache_path)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: dict,
    fcfg: dict,
    station_categories: list[str] | None = None,
    source_path: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (X, categorical_feature_names, station_categories).

    `station_categories`: if provided, the station column is converted to
    pandas Categorical with these exact categories — guarantees train/test
    consistency. If None, categories are inferred from df.

    `source_path`: optional path to the interim parquet `df` came from.
    Used as a cache key for the (expensive) solar geometry features.
    """
    ts_col = cfg["columns"]["timestamp"]
    station_col = cfg["columns"]["station"]
    target_col = cfg["columns"]["target"]

    out = df.copy()
    if fcfg["time"]["enabled"]:
        out = add_time_features(
            out, ts_col,
            include_raw=fcfg["time"]["include_raw"],
            cyclical=fcfg["time"]["cyclical"],
        )

    if fcfg.get("solar_geometry", {}).get("enabled", False):
        out = _attach_solar_features(out, cfg, source_path)

    drop_cols = {
        cfg["columns"]["id"], ts_col, target_col,
        cfg["columns"]["station_name"], cfg["columns"]["country"],
    }
    feature_cols = [c for c in out.columns if c not in drop_cols]

    cat_features: list[str] = []
    used_categories: list[str] = []
    if fcfg.get("station_features", {}).get("station_categorical", True):
        if station_categories is None:
            out[station_col] = out[station_col].astype("category")
            used_categories = list(out[station_col].cat.categories)
        else:
            out[station_col] = pd.Categorical(out[station_col], categories=station_categories)
            used_categories = list(station_categories)
        cat_features.append(station_col)
    else:
        feature_cols = [c for c in feature_cols if c != station_col]

    X = out[feature_cols].copy()
    rename = {c: sanitize_column_name(c) for c in X.columns}
    X = X.rename(columns=rename)
    cat_features = [rename[c] for c in cat_features]
    return X, cat_features, used_categories


def main(config_path: str, features_path: str) -> None:
    """Optional CLI: precompute and cache processed parquets."""
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)

    log.info("Loading interim parquets...")
    train = read_parquet(cfg["paths"]["train_interim"])
    test = read_parquet(cfg["paths"]["test_interim"])

    log.info("Building train feature matrix...")
    X_tr, _, station_cats = build_feature_matrix(train, cfg, fcfg)
    log.info("Building test feature matrix with shared station categories...")
    X_te, _, _ = build_feature_matrix(test, cfg, fcfg, station_categories=station_cats)

    write_parquet(X_tr.assign(**{cfg["columns"]["target"]: train[cfg["columns"]["target"]].values}),
                  cfg["paths"]["processed_dir"] + "/train.parquet")
    write_parquet(X_te.assign(**{cfg["columns"]["id"]: test[cfg["columns"]["id"]].values}),
                  cfg["paths"]["processed_dir"] + "/test.parquet")
    log.info("Done. train=%s test=%s", X_tr.shape, X_te.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    args = parser.parse_args()
    main(args.config, args.features)