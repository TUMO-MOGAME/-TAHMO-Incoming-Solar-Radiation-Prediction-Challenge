"""Feature pipeline orchestrator: interim → processed.

Reads `features.yaml`, applies enabled feature groups in order, and writes
processed parquet files ready for modeling.

Usage:
    python -m src.features.build_features --config config/config.yaml --features config/features.yaml
"""
from __future__ import annotations

import argparse

from src.features.time_features import add_time_features
from src.utils.io import load_config, read_parquet, write_parquet
from src.utils.logging import setup_logger


def main(config_path: str, features_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)
    ts_col = cfg["columns"]["timestamp"]

    log.info("Loading interim parquets...")
    train = read_parquet(cfg["paths"]["train_interim"])
    test = read_parquet(cfg["paths"]["test_interim"])

    if fcfg["time"]["enabled"]:
        log.info("Adding time features")
        train = add_time_features(train, ts_col, fcfg["time"]["include_raw"], fcfg["time"]["cyclical"])
        test = add_time_features(test, ts_col, fcfg["time"]["include_raw"], fcfg["time"]["cyclical"])

    # TODO: solar_geometry, weather_lags, station_features once implemented.

    train_out = cfg["paths"]["processed_dir"] + "/train.parquet"
    test_out = cfg["paths"]["processed_dir"] + "/test.parquet"
    log.info("Writing processed parquets: %s, %s", train_out, test_out)
    write_parquet(train, train_out)
    write_parquet(test, test_out)
    log.info("Done. train=%s test=%s", train.shape, test.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    args = parser.parse_args()
    main(args.config, args.features)
