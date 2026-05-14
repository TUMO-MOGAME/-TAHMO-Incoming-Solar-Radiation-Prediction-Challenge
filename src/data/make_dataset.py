"""Build interim parquet files from raw CSVs.

Usage:
    python -m src.data.make_dataset --config config/config.yaml
"""
from __future__ import annotations

import argparse

from src.data.load import read_sample_submission, read_test, read_train
from src.data.validate import validate_dataframe
from src.utils.io import load_config, write_parquet
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    cols = cfg["columns"]
    required_train = [
        cols["id"], cols["timestamp"], cols["station"], cols["station_name"],
        cols["country"], cols["target"],
        *cols["weather"], *cols["station_meta"],
    ]
    required_test = [c for c in required_train if c != cols["target"]]

    log.info("Reading raw CSVs...")
    train = read_train(cfg["paths"]["train_raw"])
    test = read_test(cfg["paths"]["test_raw"])
    sample = read_sample_submission(cfg["paths"]["sample_submission_raw"])
    log.info("train: %s, test: %s, sample: %s", train.shape, test.shape, sample.shape)

    log.info("Validating...")
    range_checks = {
        cols["target"]: (0.0, 1500.0),
        "relativehumidity (-)": (0.0, 1.0),
        "temperature (degrees Celsius)": (-20.0, 60.0),
        "precipitation (mm)": (0.0, 500.0),
    }
    for df, name, required in [
        (train, "train", required_train),
        (test, "test", required_test),
    ]:
        report = validate_dataframe(
            df, name, required, id_col=cols["id"], range_checks=range_checks
        )
        log.info(
            "%s: rows=%d cols=%d missing=%s dup_ids=%d nulls=%s issues=%s",
            report.name, report.n_rows, report.n_cols,
            report.missing_columns, report.duplicate_ids,
            report.null_counts, report.issues,
        )

    log.info("Writing interim parquet...")
    write_parquet(train, cfg["paths"]["train_interim"])
    write_parquet(test, cfg["paths"]["test_interim"])
    write_parquet(sample, cfg["paths"]["sample_submission_interim"])
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
