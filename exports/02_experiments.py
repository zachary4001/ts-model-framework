# 02 - Model Experiments


# - Version 1.01  
# - updated 30.04.26  
# 
# Pending:  
# - adding Hyperopt tuning for classical models  


# Runs all models against the same dataset and logs results to MLflow.
# All preprocessing imported from 01_preprocessing.ipynb via %run.
# 
# | Notebook | Purpose |
# |----------|---------|
# | 00_EDA | Explore dataset |
# | 01_preprocessing | Prepare model variants |
# | 02_experiments | Train, evaluate, compare (here) |


## Imports


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import sys
sys.path.append(r'Q:\scripts\projects\ts-model-framework')
import config

mlflow.set_tracking_uri(config.MLFLOW_URI)
mlflow.set_experiment(config.EXPERIMENT)

import warnings
warnings.filterwarnings('ignore')

print(f"Project path: {config.PROJECT_PATH}")
print(f"Data path: {config.DATA_PATH}")
print(f"Data path: {config.EXPORTS_PATH}")

print("Libraries loaded.")
print(f"MLflow tracking: {config.EXPERIMENT}")


# ---
## S1 - Load Preprocessed Data
# Runs 01_preprocessing.ipynb and inherits all prepared datasets.


# %run {os.getenv('PREPROCESSING_SCRIPT')}


%run Q:/scripts/projects/ts-model-framework/exports/01_preprocessing.py


# ---
## S2 - Shared Evaluation Function
# Single function used by ALL models for consistent metric calculation.


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    bias = np.mean(y_pred - y_true)

    print(f"\n{model_name}")
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ---
## S3 - XGBoost Baseline
# No hyperparameter tuning -- default parameters only.
# Goal: establish a reproducible baseline before any optimization.


xgb_params = {
    "n_estimators": 1000,
    "early_stopping_rounds": 50
}

with mlflow.start_run(run_name="xgboost-baseline"):
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

    preds_xgb = model_xgb.predict(X_test)
    metrics   = evaluate(y_test, preds_xgb, "XGBoost Baseline")
    metrics_xgb = evaluate(y_test, preds_xgb, "XGBoost Baseline")
    mlflow.log_params(xgb_params)
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model_xgb, "model")


# ---
## S4 - SARIMAX Baseline


from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_order        = (1, 0, 1)
sarimax_seasonal     = (1, 0, 0, 7)

with mlflow.start_run(run_name="sarimax-baseline"):
    model_sarimax = SARIMAX(
        y_sarimax_train,
        exog=exog_train,
        order=sarimax_order,
        seasonal_order=sarimax_seasonal
    ).fit(disp=False)

    preds_sarimax = model_sarimax.forecast(
        steps=len(y_sarimax_test),
        exog=exog_test
    )
    metrics = evaluate(y_sarimax_test.values, preds_sarimax.values, "SARIMAX Baseline")
    metrics_sarimax = evaluate(y_sarimax_test.values, preds_sarimax.values, "SARIMAX Baseline")

    mlflow.log_params({"order": str(sarimax_order), "seasonal_order": str(sarimax_seasonal)})
    mlflow.log_metrics(metrics)


# ---
## S5 - Prophet Baseline


from prophet import Prophet

with mlflow.start_run(run_name="prophet-baseline"):
    model_prophet = Prophet()
    model_prophet.fit(prophet_train)

    future   = model_prophet.make_future_dataframe(periods=len(prophet_test))
    forecast  = model_prophet.predict(future)
    preds_prophet = forecast['yhat'].tail(len(prophet_test)).values

    metrics = evaluate(prophet_test['y'].values, preds_prophet, "Prophet Baseline")
    metrics_prophet = evaluate(prophet_test['y'].values, preds_prophet, "Prophet Baseline")
    
    mlflow.log_params({"model": "prophet-basic"})
    mlflow.log_metrics(metrics)


# ---
## S6 - Results Summary
# Compares all logged runs from this experiment in a single table.


runs = mlflow.search_runs(experiment_names=["ts-model-framework"])

summary = runs[["tags.mlflow.runName", "metrics.rmse", "metrics.mae", 
                "metrics.mape", "metrics.r2"]].rename(
    columns={"tags.mlflow.runName": "model"}
).sort_values("metrics.rmse")

print(summary.to_string(index=False))


# ---
## S7 - Visual Comparison


fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
ax.plot(y_test.index, preds_xgb, label='XGBoost', linestyle='--')
ax.plot(y_sarimax_test.index, preds_sarimax, label='SARIMAX', linestyle='--')
ax.plot(prophet_test['ds'].values[-len(preds_prophet):], 
        preds_prophet, label='Prophet', linestyle='--')

ax.set_title('Model Comparison -- Test Period Forecasts vs Actuals')
ax.set_ylabel(config.TARGET_COLUMN)
ax.legend()
plt.tight_layout()
plt.show()


# ---  
# 
## S8 -  Save Best Model


# Saves the best model from Classical + XGBoost only

# Collect all model results
model_objects = {
    "xgboost-baseline": model_xgb,
    "sarimax-baseline": model_sarimax,
    "prophet-baseline": model_prophet,
}
# Pick best from LOCAL results only -- not MLflow
local_results = {
    "xgboost-baseline": metrics_xgb["rmse"],
    "sarimax-baseline": metrics_sarimax["rmse"],
    "prophet-baseline": metrics_prophet["rmse"],
}
best_model_name = min(local_results, key=local_results.get)
print(f"Best classical/XGBoost model: {best_model_name}")

# Save best model for Streamlit
# best_model_object = model_objects[best_model_name]
save_path = os.path.join(config.MODELS_PATH, "best_model.pkl")
joblib.dump(model_objects[best_model_name], save_path)

name_path = os.path.join(config.MODELS_PATH, "best_model_name.txt")
with open(name_path, "w") as f:
    f.write(best_model_name)

print(f"Model saved: {save_path}")
print(f"Model name saved: {name_path}")


# ---
## S9 - Notes & Observations
# Document findings per experiment run.
# 
# - Best model this run:
# - Notable differences between models:
# - Features that helped/hurt:
# - Next experiment to try:


