# TAHMO Incoming Solar Radiation Prediction Challenge

Predict incoming solar radiation (W/m²) at 15-minute intervals for withheld even months of Year 1 across 40 TAHMO weather stations in Africa.

## Quick start

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build interim dataset from raw CSVs
python -m src.data.make_dataset

# 4. Run baseline cross-validation
python -m src.cv.run_cv --config config/config.yaml

# 5. Train final model and generate submission
python -m src.train --config config/config.yaml
python -m src.predict --config config/config.yaml
python -m src.submit --config config/config.yaml
```

## Project layout

```
config/         YAML configuration (paths, features, model)
data/
  raw/          Original competition CSVs (gitignored)
  interim/      Cleaned, typed, sorted (parquet)
  processed/    Feature-engineered, ready for modeling
src/
  data/         Loading and validation
  features/     Time / solar geometry / weather / station features
  cv/           Cross-validation splitters and metric evaluation
  models/       Model wrappers (LightGBM, XGBoost, CatBoost, ensembles)
  utils/        Seeding, IO, logging
notebooks/      EDA and exploration only (not pipeline logic)
experiments/    Per-experiment configs, CV scores, OOF predictions
submissions/    Generated submission CSVs
```

## Metric

The competition scores both **MBE** (Mean Bias Error) and **RMSE** of predicted radiation against hidden observations. The submission file therefore has two columns (`TargetMBE`, `TargetRMSE`) — see `src/cv/evaluate.py` for the implementation and `CLAUDE.md` for current thinking on whether to submit the same prediction in both.

## Reproducing a previous submission

Each submission CSV has a sibling experiment folder under `experiments/exp_NNN_description/` containing the config and code state used to produce it. The final polished notebook (`notebooks/reproduce_winning_solution.ipynb`) is built at the end of the project.

## Project context for Claude

`CLAUDE.md` is the persistent context for AI sessions — it tracks current state, what's been tried, and what's queued. Keep it updated as work progresses.