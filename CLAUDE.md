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

**Key facts (confirmed by EDA on 2026-05-14):**
- 40 stations, all appearing in both train and test
- Train date range: 2016-01-11 → 2020-11-30
- Test date range: 2016-02-01 → 2020-12-31
- **Train contains only ODD months {1, 3, 5, 7, 9, 11}. Test contains only EVEN months {2, 4, 6, 8, 10, 12}.** This is the central challenge — the model must predict months it has never seen.
- The earlier reading of "withheld even months of Year 1" was misleading; the split is global across all 2016-2020 years.
- 39/40 stations have full coverage of all 6 odd months in train; TA00118 has effectively no January data.
- Target is non-negative; observed max = 1427 W/m²; **50.3% of training rows have target == 0** (nighttime).
- No missing values in the core columns.
- `ID` format: `{stationhash}_{YYYY-MM}_{random6}` — month embedded in the ID matches the timestamp.

**Use only this data.** No external EO/reanalysis data permitted in our approach (user decision).

---

## 3. Metric

The competition (per Zindi page, confirmed 2026-05-14) computes:

- **|MBE|** = `|mean(y_pred - y_true)|` — absolute, not signed.
- **RMSE** = `sqrt(mean((y_pred - y_true)^2))`.
- **LB score = 0.5 * |MBE| + 0.5 * RMSE** (this is `combined_score` in `src/cv/evaluate.py`).

**Submission format requirement**: 3 columns `ID, TargetMBE, TargetRMSE` with **identical values in TargetMBE and TargetRMSE per row**. Putting different values in the two columns is not allowed — the MBE-vs-RMSE column investigation is closed.

**LB vs CV scale**: Zindi normalizes the LB score (e.g., baseline CV combined 50.70 in W/m² maps to LB 0.099). Normalization is undocumented but constant, so relative improvements in CV should track relative improvements on LB.

---

## 4. Cross-validation strategy

The test set is the **even months** while train is the **odd months** of every year. Train has zero data in months 2, 4, 6, 8, 10, 12 — so we cannot hold out even months. Instead, we simulate the "predict an unseen month" challenge by leave-one-month-out over the 6 odd months.

**Primary CV** (`src/cv/splitters.py: LeaveOneMonthOutSplitter`):
6 folds. For each fold, hold out one odd month from train, fit on the other five, score MBE+RMSE on the held-out month. Each fold's validation is the closest analogue to the test challenge we can build from train alone.

**Secondary CV** (`src/cv/splitters.py: TimeForwardSplitter`):
Per-station forward-walking time splits — sanity check for temporal leakage.

**Never** use random K-fold over rows — radiation is highly autocorrelated within a station-day; random splits leak future into the past.

---

## 5. Pipeline stages

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data ingestion | `src/data/load.py`, `validate.py`, `make_dataset.py` | Done |
| 2. EDA | inline scripts (no notebook until end) | Done — findings in §2, §8 |
| 3. CV scaffolding | `src/cv/splitters.py`, `evaluate.py` | Done |
| 4. Baseline LightGBM (time features only) | `src/cv/run_cv.py`, `src/models/lgbm.py` | Done — exp_001 |
| 5. Time features | `src/features/time_features.py` | Done |
| 6. Solar geometry features | `src/features/solar_geometry.py` | Skeleton — NEXT priority |
| 7. Weather lag/rolling features | `src/features/weather_features.py` | Skeleton |
| 8. Station-level features | `src/features/station_features.py` | Skeleton |
| 9. XGBoost model | `src/models/xgb.py` | Skeleton |
| 10. CatBoost model | `src/models/catboost.py` | Skeleton |
| 11. Ensemble | `src/models/ensemble.py` | Skeleton |
| 12. MBE-vs-RMSE column investigation | `src/submit.py` | TODO |
| 13. Final submission | `src/train.py`, `predict.py`, `submit.py` | Skeleton |
| 14. Reproduction notebook | `notebooks/reproduce_winning_solution.ipynb` | Built last |

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

- Project fully scaffolded: configs, data ingestion, CV splitter, evaluation, LightGBM wrapper, CV runner all functional.
- **EDA confirms train = odd months / test = even months globally.** CV splitter rewritten as `LeaveOneMonthOutSplitter` over the 6 odd months.
- **First baseline trained**: single LightGBM with time features + station-as-categorical. Score below.
- Original starter notebook preserved at `notebooks/tahmo_starter_notebook.ipynb`.

### Leaderboard log

| Exp # | Description | CV \|MBE\| | CV RMSE | CV combined | LB (public) | Notes |
|-------|-------------|--------|---------|-------------|-------------|-------|
| exp_001_lgbm_baseline | LightGBM, time features + station categorical, 6-fold LOMO | 1.07 | 100.33 | 50.70 | **0.099254** | per-fold MBE swings ±20, Jan fold worst (RMSE 114.9); LB-vs-CV scale factor ≈ 0.00196 |

### Feature importance (exp_001)

Top 5 features = 84% of total gain. Hour-of-day (`hour_cos`, `hour`, `hour_sin`) alone is ~75%. `temperature` (#3), `station` (#4), `latitude`/`longitude` follow. **`month` and cyclical month features are weak** — the model is correctly distrustful of features that take values it has never seen in train. This is a strong signal that solar-geometry features (zenith angle, theoretical clearsky GHI) should improve scores substantially.

### What's queued next (priority order)

1. **Solar geometry features via pvlib** — solar zenith, azimuth, clearsky GHI, day_length, is_daylight. Expected biggest single jump because hour-of-day is currently doing all this work implicitly.
2. **Weather lag/rolling features** — humidity and temperature lags within station-day.
3. **Per-station bias correction** — per-station mean residual subtraction; targets the |MBE| half of the score directly.
4. **XGBoost and CatBoost** for model diversity, then ensemble.
5. **LightGBM hyperparameter tuning** — last lever once features and ensemble are exhausted.

(MBE-vs-RMSE column trick is removed — Zindi mandates identical values per row.)

### Decisions made

- **Single global LightGBM**, with `station` as categorical — not per-station models. Easier to validate, more signal per fit.
- **LightGBM 4.6.0** as primary. Categorical feature names sanitized to remove special chars via `sanitize_column_name` in `src/cv/run_cv.py`.
- **Parquet for interim/processed data** — 5× faster reads than CSV.
- **`clip_max = 1500 W/m²`** based on observed train max 1427.

### Things to investigate

- Per-station score breakdown (`experiments/exp_001_lgbm_baseline/per_station_score.csv`) — are some stations systematically worse? They may need station-specific bias correction.
- Whether `TargetMBE` and `TargetRMSE` accept different predictions for leaderboard advantage.
- TA00118 has minimal January data — does it have a unique failure mode in CV?

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