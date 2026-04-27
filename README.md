# Time Series Model Comparison Framework

A structured, reusable framework for comparing time series forecasting models
on any dataset. Built during Masterschool MSIT Term 8 (Time Series Modeling).

## Structure
- `data/`      — raw and processed datasets
- `notebooks/` — sandbox notebooks (build and test here)
- `exports/`   — stable .py exports (imported by other notebooks)
- `mlflow-artifacts/` — local MLflow artifact storage

## Workflow
1. EDA and preprocessing in notebooks
2. Export stable functions to exports/
3. Run experiments, log everything to MLflow
4. Compare models in MLflow UI (localhost:5000)

## Models Covered
- SARIMAX
- Holt-Winters
- Prophet
- XGBoost
- RNN / LSTM (planned)

## Requirements
See requirements.txt