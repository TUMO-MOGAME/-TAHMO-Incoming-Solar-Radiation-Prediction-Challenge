# CLAUDE.md — Project Context for AI Sessions

This file is auto-loaded by Claude Code on every session. It is the **single source of truth** for what this project is, what's been done, and what's queued next. Keep it updated as work progresses.

---

## 1. Project goal

**TAHMO Incoming Solar Radiation Drift Correction Challenge.**
Predict `radiation (W/m²)` at 15-minute intervals for **withheld even months of Year 1** for each of 40 TAHMO weather stations across Africa.

Submitted on Zindi. We are aiming for the top of the leaderboard.

---

## 2. Data

All raw competition files live in `data/raw/` (gitignored — too large).

| File | Rows | Cols | Notes |
|------|------|------|-------|
| `Train.csv` | 642,175 | 13 | Includes the target `radiation (W/m²)` |
| `Test.csv` | 683,353 | 12 | Same schema minus target |
| `SampleSubmission.csv` | 683,353 | 3 | `ID`, `TargetMBE`, `TargetRMSE` |
| `dataset_data_dictionary.csv` | 13 | 2 | Column descriptions (kept in git) |

**Columns** (Train.csv): `ID`, `timestamp`, `precipitation (mm)`, `radiation (W/m²)` *(target)*, `relativehumidity (-)`, `temperature (degrees Celsius)`, `station`, `station_name`, `country`, `installation_height`, `elevation`, `latitude`, `longitude`.

**Key facts:**
- 40 stations, all appearing in both train and test
- Train date range: 2016-01-11 → 2020-11-30
- Test date range: 2016-02-01 → 2020-12-31
- No missing values in the core columns
- `ID` format: `{stationhash}_{YYYY-MM}_{random6}` — month is embedded in the ID
- Target is non-negative; large fraction of zeros (nighttime)

**Use only this data.** No external EO/reanalysis data permitted in our approach (user decision).

---

## 3. Metric

The competition computes **MBE** (Mean Bias Error) and **RMSE** between predicted and hidden observed radiation:

- `MBE = mean(y_pred - y_true)` — signed; positive = overprediction
- `RMSE = sqrt(mean((y_pred - y_true)^2))`

The submission file has two columns (`TargetMBE`, `TargetRMSE`). The baseline puts the same prediction in both — investigate whether a bias-adjusted prediction in `TargetMBE` and the raw prediction in `TargetRMSE` improves leaderboard score.

Our combined CV score: see `src/cv/evaluate.py`.

---

## 4. Cross-validation strategy

The test set is the **withheld even months of Year 1** per station. Our CV must mimic this.

**Primary CV** (`src/cv/splitters.py: EvenMonthHoldoutSplitter`):
Hold out one or more even months from train; train on the rest; score MBE+RMSE on the held-out portion. Rotate which even months are held out across folds.

**Secondary CV** (`src/cv/splitters.py: TimeForwardSplitter`):
Per-station forward-walking time splits — sanity check for temporal leakage.

**Never** use random K-fold over rows — radiation is highly autocorrelated within a station-day; random splits leak future into the past.

---

## 5. Pipeline stages

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data ingestion | `src/data/load.py`, `validate.py`, `make_dataset.py` | Scaffolded |
| 2. EDA | `notebooks/01_eda.ipynb` | TODO |
| 3. CV scaffolding | `src/cv/splitters.py`, `evaluate.py` | Scaffolded |
| 4. Baseline reproduction | `src/cv/run_cv.py` | TODO |
| 5. Time features | `src/features/time_features.py` | Skeleton |
| 6. Solar geometry features | `src/features/solar_geometry.py` | Skeleton |
| 7. Weather lag/rolling features | `src/features/weather_features.py` | Skeleton |
| 8. Station-level features | `src/features/station_features.py` | Skeleton |
| 9. LightGBM model | `src/models/lgbm.py` | Skeleton |
| 10. XGBoost model | `src/models/xgb.py` | Skeleton |
| 11. CatBoost model | `src/models/catboost.py` | Skeleton |
| 12. Ensemble | `src/models/ensemble.py` | Skeleton |
| 13. MBE-vs-RMSE column investigation | `src/submit.py` | TODO |
| 14. Final submission | `src/train.py`, `predict.py`, `submit.py` | Skeleton |
| 15. Reproduction notebook | `notebooks/reproduce_winning_solution.ipynb` | Built last |

---

## 6. Conventions

- **Configs over hardcoding.** Paths, seeds, column names, hyperparameters all live in `config/*.yaml`.
- **One experiment = one folder** under `experiments/exp_NNN_description/` with its config snapshot, CV scores, and OOF predictions.
- **Save interim data as parquet** (`data/interim/train.parquet`, etc.) — 10× faster than re-parsing CSV every run.
- **Seeds are set** via `src.utils.seed.set_seed(seed)` in every entry point.
- **Logging via stdlib `logging`**, configured by `src.utils.logging.setup_logger()`.
- **Type hints + brief docstrings** on every public function. No multi-paragraph docstrings.
- **Never hardcode column names** in feature/model code — pull from `config.yaml`.

---

## 7. How to run things

```bash
# Build the interim parquet files from raw CSVs (one-time, ~30s)
python -m src.data.make_dataset

# Run a full CV experiment with a given config
python -m src.cv.run_cv --config config/config.yaml

# Train final model on all data, generate test predictions, build submission
python -m src.train --config config/config.yaml
python -m src.predict --config config/config.yaml
python -m src.submit --config config/config.yaml
```

---

## 8. Current state

**Date:** 2026-05-14

- Project scaffolded: folder structure, configs, data loading, CV splitter, evaluation metric.
- Baseline starter notebook is preserved at `notebooks/tahmo_starter_notebook.ipynb` for reference.
- **No models trained yet.** Next: implement the LightGBM baseline + run first CV.

### Leaderboard log

| Exp # | Description | CV MBE | CV RMSE | CV combined | LB MBE | LB RMSE | Notes |
|-------|-------------|--------|---------|-------------|--------|---------|-------|
| _none yet_ | | | | | | | |

### What's queued next

1. Implement `src/data/load.py` and run `make_dataset.py` to generate interim parquet.
2. Implement `EvenMonthHoldoutSplitter` and confirm it splits sensibly per station.
3. Implement LightGBM baseline (`src/models/lgbm.py`), run first CV.
4. Add solar geometry features (`pvlib`-based) and re-score.

### Decisions made

- **Global single-model first**, with `station` as categorical — not per-station models like the starter. Easier to validate, more signal per fit.
- **LightGBM as primary** — handles categoricals natively, fast, robust.
- **Parquet for interim/processed data** — speed and storage win.

### Things to investigate

- Is `Year 1` the first calendar year per station, or 2017 globally? Check via `ID` parsing.
- Distribution of even-month coverage per station in train (is it uniform?).
- Whether `TargetMBE` and `TargetRMSE` accept different predictions for leaderboard advantage.

---

## 9. File map (cheat sheet)

```
config/config.yaml              # paths, ids, seeds
config/features.yaml            # feature group toggles
config/model.yaml               # model class + hyperparams
src/data/load.py                # read_train, read_test, read_sample_submission
src/data/validate.py            # check_schema, check_ranges
src/data/make_dataset.py        # raw → interim parquet
src/features/time_features.py   # cyclical hour/month/doy
src/features/solar_geometry.py  # zenith, azimuth, day_length, clearsky_ghi
src/features/weather_features.py # lag/rolling humidity, temp, precip
src/features/station_features.py # CV-safe station target encoding
src/features/build_features.py  # orchestrator
src/cv/splitters.py             # EvenMonthHoldoutSplitter, TimeForwardSplitter
src/cv/evaluate.py              # mbe, rmse, combined_score
src/cv/run_cv.py                # full CV loop, writes experiments/exp_*/
src/models/base.py              # shared model interface
src/models/{lgbm,xgb,catboost}.py
src/models/ensemble.py
src/train.py                    # train final on all data
src/predict.py                  # produce test predictions
src/submit.py                   # build SampleSubmission-shaped CSV
src/utils/{seed,io,logging}.py
```