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

## 3. Metric — Reverse-engineered from 3 submissions (2026-05-14)

Zindi's docs describe a "weighted mean of MBE and RMSE with 0.5/0.5 weights" but reality is different. By fitting three submission outcomes we get:

**Public Score ≈ 1 − 0.0843 · AbsMBE − 0.00545 · RMSE**

Verified against 3 submissions:
- baseline:  1 − 0.0843·4.98 − 0.00545·88.08 = 0.100 (actual 0.0993)
- solar corrected: 1 − 0.0843·6.06 − 0.00545·86.62 = 0.017 (actual 0.0163)
- solar raw: 1 − 0.0843·44.59 − 0.00545·326.05 = −4.536 (actual −4.5364)

**Per unit, AbsMBE is 15.5× more valuable to fix than RMSE.** This is the most important fact about the competition.

**Note on "AbsMBE"**: it is *not* simple global \|mean(pred − true)\|. A constant +3.32 shift on all predictions changed AbsMBE from 44.59 → 6.06 — far more than the 3.32 expected from a global shift. Most likely Zindi computes per-station signed MBE then averages the absolute values (so a global shift interacts with per-station sign patterns), or some similar per-group aggregation. **This means per-station bias correction is dramatically more valuable than a single global shift.**

**Submission format**: 3 columns `ID, TargetMBE, TargetRMSE` with **identical values per row** (experimentally confirmed). A probe submission with non-identical columns was rejected. Always write the same value to both columns. Column-split investigation closed for good.

---

## 4. Cross-validation strategy

The test set is the **even months** while train is the **odd months** of every year. Train has zero data in months 2, 4, 6, 8, 10, 12 — so we cannot hold out even months. Instead, we simulate the "predict an unseen month" challenge by leave-one-month-out over the 6 odd months.

Every experiment runs **three** CV strategies and produces `cv_dashboard.{json,txt}` for triangulation. Implementations in `src/cv/splitters.py`.

| # | Strategy | Role | What it tells us |
|---|----------|------|------------------|
| 1 | **LeaveOneMonthOutSplitter** | PRIMARY (decisions) | Direct simulation of the test task: predict an unseen month from the other 5 odd months. 6 folds. |
| 2 | **GroupKFoldByYearSplitter** | Sanity check | Hold out one calendar year. 5 folds (2016-2020). Catches over-fit to 2018 (68% of train rows). |
| 3 | **TimeForwardSplitter** | Sanity check | Per-station forward walk. If this scores better than primary, a feature is leaking future info. |

**Interpretation rules** (auto-computed in `cv_dashboard.txt`, lower combined = better):

- All three should move down together as features improve → trustworthy improvement.
- **TimeForward < 0.85 × Primary** → 🚩 feature leakage; some signal is from the future.
- **GroupKFoldByYear > 1.20 × Primary** → 🚩 year-overfit; model leans too hard on year-specific patterns. Consider dropping the `year` feature, sample-weighting under-represented years, or stronger regularization.
- One strategy moves but the others don't → suspicious, investigate before celebrating.

**Strategies we deliberately do NOT use** (would mislead on this data):

- Random `KFold` — radiation is autocorrelated within a station-day at 15-min lag, so random splits leak across consecutive rows and produce wildly optimistic scores.
- `StratifiedKFold` — classification tool; our challenge is temporal generalization, not class imbalance.
- `GroupKFold` by station — all 40 stations are in BOTH train and test; holding out stations tests the wrong question.
- `StratifiedGroupKFold` — every station appears in every month, so there's no useful axis to stratify on.

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

| Exp # | Description | LOMO combined (PRIMARY) | GroupKFoldByYear combined | TimeForward combined | LB (public) | Notes |
|-------|-------------|--------|---------|-------------|-------------|-------|
| exp_001_lgbm_baseline | LightGBM, time features + station categorical | **50.70** (CV \|MBE\|=1.07, RMSE=100.33) | 77.64 (ratio 1.53 — 🚩 year overfit) | 50.57 (ratio 1.00) | **0.099254** | LB AbsMBE=4.98, RMSE=88.08. CV pooled \|MBE\| was 5× too optimistic. CV RMSE was 12 W/m² too pessimistic. |
| exp_002_lgbm_solar | Adds pvlib solar geometry | 51.26 (CV \|MBE\|=3.32, RMSE=99.20) | 65.87 (year ratio 1.28) | 51.01 | **−4.536** raw / **0.016** corrected | DISASTER on LB. Raw AbsMBE blew up to 44.59, RMSE to 326. Solar features destabilize per-station predictions; CV does not catch this. Bias correction recovered AbsMBE to 6.06 but score still below baseline. Solar features are not viable as-is. |
| submission_003_baseline_per_station_corrected | exp_001 predictions − per-station OOF MBE | (no CV — bias correction step) | — | — | pending upload | If per-station correction works as expected, predicted LB ≈ 0.40–0.52 (vs baseline 0.099). Big bet — first real test of the "AbsMBE is per-station aggregated" hypothesis. |

### Feature importance (exp_001)

Top 5 features = 84% of total gain. Hour-of-day (`hour_cos`, `hour`, `hour_sin`) alone is ~75%. `temperature` (#3), `station` (#4), `latitude`/`longitude` follow. **`month` and cyclical month features are weak** — the model is correctly distrustful of features that take values it has never seen in train. This is a strong signal that solar-geometry features (zenith angle, theoretical clearsky GHI) should improve scores substantially.

### What's queued next (priority order)

1. **Upload submission_003** (per-station bias-corrected baseline). Predicted LB jump from 0.099 → 0.4–0.5. If this works, per-station bias correction becomes a permanent step in the pipeline.
2. **If #1 works**: refine the per-station MBE estimate. Use per-fold per-station signed MBE instead of just OOF aggregate (more robust against fold-wise sign cancellation). Submit as 003b.
3. **Custom training objective**: train LightGBM with an objective that directly approximates `0.0843·|MBE| + 0.00545·RMSE`. The default `regression` objective optimizes pure squared error, which is 15× under-weighted relative to what the LB cares about. A custom objective could give big gains.
4. **Weather lag/rolling features** — captures cloud-cover transients that the model currently can't see.
5. **XGBoost and CatBoost** for model diversity, then ensemble.
6. **LightGBM hyperparameter tuning** — last lever once features and ensemble are exhausted.

(Solar features are paused — they destabilize per-station predictions in a way CV does not capture. Possibly revisit with elevation-restricted clearsky model or African-tuned Linke turbidity, but not a priority.)

### Decisions made

- **Single global LightGBM**, with `station` as categorical — not per-station models. Easier to validate, more signal per fit.
- **LightGBM 4.6.0** as primary. Categorical feature names sanitized to remove special chars via `sanitize_column_name` in `src/features/build_features.py`.
- **Parquet for interim/processed data** — 5× faster reads than CSV.
- **`clip_max = 1500 W/m²`** based on observed train max 1427.
- **Solar geometry features are cached** to `data/processed/{train,test}_solar.parquet` keyed by source interim parquet. First compute ~8s; subsequent runs reuse.
- **Bias correction at submission time**: when the OOF mean signed residual is non-trivial (>1 W/m²), add `-mean(residual)` to all test predictions before clipping. Targets the \|MBE\| half of the score with no model retraining.

### Things to investigate

- **Year-overfit flag** (GroupKFoldByYear ratio 1.53 in exp_001): the baseline leans on year-specific patterns. Worst single-year fold was 2016 (combined 143.1). Candidates: drop the raw `year` feature, sample-weight under-represented years (2016/2019/2020), or stronger regularization. Check whether solar geometry features close this gap before adding mitigations — physics features are year-invariant by construction.
- Per-station score breakdown (`experiments/exp_001_lgbm_baseline/per_station_score.csv`) — are some stations systematically worse? They may need station-specific bias correction.
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