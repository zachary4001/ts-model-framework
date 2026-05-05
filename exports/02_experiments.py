# 02 - Model Experiments

# - Version 1.05 
# - updated 05.05.26  

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
## S2 - Oil Correlation Analysis
# 


# Oil Correlation Analysis

# Fill forward zeros
df_full = pd.concat([df_train, df_test]).sort_index()
df_full['dcoilwtico'] = df_full['dcoilwtico'].replace(0, np.nan).ffill()

# set chart parameters
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# --- Plot 1: Dual-axis overlay ---
ax1 = axes[0]
ax2 = ax1.twinx()

df_full = pd.concat([df_train, df_test]).sort_index()

ax1.plot(df_full.index, df_full['unit_sales'], color='steelblue', 
         label='Unit Sales', linewidth=1.2)
ax2.plot(df_full.index, df_full['dcoilwtico'], color='darkorange', 
         alpha=0.7, label='Oil Price (WTI)', linewidth=1.2)

ax1.set_ylabel('Unit Sales', color='steelblue')
ax2.set_ylabel('Oil Price USD', color='darkorange')
ax1.set_title('Unit Sales vs Oil Price -- Full Period')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# --- Plot 2: Rolling 30-day correlation ---
df_corr = df_full[['unit_sales', 'dcoilwtico']].ffill()
rolling_corr = df_corr['unit_sales'].rolling(30).corr(df_corr['dcoilwtico'])

axes[1].plot(df_full.index, rolling_corr, color='purple', linewidth=1.2)
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Rolling Correlation (30d)')
axes[1].set_title('Rolling 30-Day Correlation: Unit Sales vs Oil Price')
axes[1].set_ylim(-1, 1)

# --- Plot 3: Scatter ---
axes[2].scatter(df_corr['dcoilwtico'], df_corr['unit_sales'], 
                alpha=0.4, color='teal', edgecolors='none', s=20)
axes[2].set_xlabel('Oil Price (WTI)')
axes[2].set_ylabel('Unit Sales')
axes[2].set_title('Scatter: Oil Price vs Unit Sales')

overall_corr = df_corr['unit_sales'].corr(df_corr['dcoilwtico'])
axes[2].annotate(f'Overall Pearson r = {overall_corr:.3f}', 
                 xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11,
                 color='darkred')

plt.tight_layout()
plt.savefig(f"{config.MODELS_PATH}/oil_correlation_analysis.png", 
            dpi=150, bbox_inches='tight')
plt.show()
print(f"Overall Pearson correlation (oil vs sales): {overall_corr:.4f}")


# S2b - Oil vs Sales Relationship (Rescaled)
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

df_plot = pd.concat([df_train, df_test]).sort_index().copy()
df_plot['dcoilwtico'] = df_plot['dcoilwtico'].replace(0, np.nan).ffill()
df_plot['oil_zscore']     = (df_plot['dcoilwtico'] - df_plot['dcoilwtico'].mean()) / df_plot['dcoilwtico'].std()
df_plot['oil_pct_change'] = df_plot['dcoilwtico'].pct_change()
df_plot['oil_above_100']  = (df_plot['dcoilwtico'] > 100).astype(int)

# --- Plot 1: Z-score overlay (same scale comparison) ---
ax1 = axes[0]
ax2 = ax1.twinx()
ax1.plot(df_plot.index, df_plot['unit_sales'], color='steelblue', 
         label='Unit Sales', linewidth=1.0, alpha=0.8)
ax2.plot(df_plot.index, df_plot['oil_zscore'], color='darkorange', 
         label='Oil Z-Score', linewidth=1.2)
ax2.axhline(0, color='darkorange', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.set_title('Unit Sales vs Oil Price Z-Score')
ax1.set_ylabel('Unit Sales', color='steelblue')
ax2.set_ylabel('Oil Z-Score', color='darkorange')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# --- Plot 2: Sales distribution -- oil above vs below $100 ---
above = df_plot[df_plot['oil_above_100'] == 1]['unit_sales']
below = df_plot[df_plot['oil_above_100'] == 0]['unit_sales']
axes[1].hist(below, bins=30, alpha=0.6, color='steelblue', label='Oil < $100 (n={})'.format(len(below)))
axes[1].hist(above, bins=30, alpha=0.6, color='darkorange', label='Oil > $100 (n={})'.format(len(above)))
axes[1].axvline(below.mean(), color='steelblue', linestyle='--', linewidth=1.5,
                label=f'Mean below: {below.mean():.0f}')
axes[1].axvline(above.mean(), color='darkorange', linestyle='--', linewidth=1.5,
                label=f'Mean above: {above.mean():.0f}')
axes[1].set_title('Unit Sales Distribution: Oil Above vs Below $100')
axes[1].set_xlabel('Unit Sales')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# --- Plot 3: Oil pct change vs next-day sales change ---
df_plot['sales_pct_change'] = df_plot['unit_sales'].pct_change()
df_clean = df_plot[['oil_pct_change', 'sales_pct_change']].replace([np.inf, -np.inf], np.nan).dropna()
axes[2].scatter(df_clean['oil_pct_change'], df_clean['sales_pct_change'],
                alpha=0.4, color='teal', s=20, edgecolors='none')
r = df_clean['oil_pct_change'].corr(df_clean['sales_pct_change'])
axes[2].axhline(0, color='black', linestyle='--', linewidth=0.6)
axes[2].axvline(0, color='black', linestyle='--', linewidth=0.6)
axes[2].set_title('Daily % Change: Oil vs Sales')
axes[2].set_xlabel('Oil Price % Change')
axes[2].set_ylabel('Unit Sales % Change')
axes[2].annotate(f'Pearson r = {r:.3f}', xy=(0.05, 0.92), 
                 xycoords='axes fraction', fontsize=11, color='darkred')

plt.tight_layout()
plt.savefig(f"{config.MODELS_PATH}/oil_sales_rescaled.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Mean sales -- oil below $100: {below.mean():.1f} | oil above $100: {above.mean():.1f}")
print(f"Daily pct change correlation: {r:.4f}")


# ---
## S2 - Shared Evaluation Function
# Single function used by ALL models for consistent metric calculation.


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = np.asarray(y_true) != 0
    mape = np.mean(np.abs((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask]) / np.asarray(y_true)[mask])) * 100
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
# 
## S4 - XGBoost Feature Ablation


# XGBoost Feature Ablation
feature_sets = {
    "xgb-calendar-only": [
        'year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year',
        'is_weekend', 'is_month_start', 'is_month_end'
    ],
    "xgb-lag-rolling-only": [
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'rolling_7d_mean', 'rolling_14d_mean', 'rolling_30d_mean', 'rolling_7d_std'
    ],
    "xgb-exogenous-only": [
        'dcoilwtico', 'oil_lag_1', 'oil_rolling_7d_mean',
        'is_national_holiday', 'is_regional_holiday', 'is_local_holiday'
    ],
    "xgb-no-exogenous": [
        'year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year',
        'is_weekend', 'is_month_start', 'is_month_end',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'rolling_7d_mean', 'rolling_14d_mean', 'rolling_30d_mean', 'rolling_7d_std'
    ],
}

for run_name, features in feature_sets.items():
    xgb_abl = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, verbosity=0)
    
    X_tr_abl = df_train[features].fillna(0)
    X_te_abl = df_test[features].fillna(0)

    with mlflow.start_run(run_name=run_name):
        xgb_abl.fit(X_tr_abl, y_train, eval_set=[(X_te_abl, y_test)], verbose=False)
        preds_abl = xgb_abl.predict(X_te_abl)
        metrics_abl = evaluate(y_test, preds_abl, run_name)
        mlflow.log_params({
            "feature_group": run_name,
            "n_features": len(features),
            "feature_list": ", ".join(features),
            "n_estimators": 1000,
            "tuned": False
})
        mlflow.log_metrics(metrics_abl)


# ---  
# 
## S4 - XGBoost Feature Importance


# XGBoost Feature Importance All
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

xgb.plot_importance(
    model_xgb,
    ax=ax,
    max_num_features=20,
    importance_type='gain',
    title='XGBoost Feature Importance (Gain)',
    xlabel='Average Gain',
    show_values=False
)

plt.tight_layout()
plt.savefig(f"{config.MODELS_PATH}/xgb_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved.")


# XGBoost Feature Importance - Top 11
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

xgb.plot_importance(
    model_xgb,
    ax=ax,
    max_num_features=11,
    importance_type='gain',
    title='XGBoost Feature Importance (Gain)',
    xlabel='Average Gain',
    show_values=False
)

plt.tight_layout()
plt.savefig(f"{config.MODELS_PATH}/xgb_feature_importance-1.png", dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved.")


# ---  
# 
## S4 - XGBoost Feature Focus


# XGBoost Feature Focused
feature_sets = {
    "xgb-minimal-day-lag1": ['dayofweek','lag_1'
    ],
    "xgb-minimal-D-W-Q-lag1": ['dayofweek','lag_1','quarter', 'week_of_year'
    ],
    "xgb-minimal-calendar": [
        'year', 'month', 'dayofweek', 'quarter', 'week_of_year',
        'is_weekend', 'is_month_start', 'lag_1'        
    ],
    "xgb-top11-features": [
        'dayofweek', 'lag_1', 'rolling_30d_mean', 'is_national_holiday', 
        'year', 'lag_30', 'week_of_year', 'is_local_holiday', 'rolling_14d_mean',
        'lag_7', 'lag_14'         
    ],
}

for run_name, features in feature_sets.items():
    xgb_abl = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, verbosity=0)
    
    X_tr_abl = df_train[features].fillna(0)
    X_te_abl = df_test[features].fillna(0)

    with mlflow.start_run(run_name=run_name):
        xgb_abl.fit(X_tr_abl, y_train, eval_set=[(X_te_abl, y_test)], verbose=False)
        preds_abl = xgb_abl.predict(X_te_abl)
        metrics_abl = evaluate(y_test, preds_abl, run_name)
        mlflow.log_params({
            "feature_group": run_name,
            "n_features": len(features),
            "feature_list": ", ".join(features),
            "n_estimators": 1000,
            "tuned": False
})
        mlflow.log_metrics(metrics_abl)


# ---
## S5 - SARIMAX Baseline


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
# 
## S6 - SARIMAX Grid Search - Baseline


# SARIMAX Grid Search on Baseline feature set
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX

p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

best_rmse, best_order, best_sarimax = float('inf'), None, None

for p, d, q in product(p_values, d_values, q_values):
    try:
        m = SARIMAX(y_sarimax_train, exog=exog_train, order=(p, d, q),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        preds = res.forecast(steps=len(y_sarimax_test), exog=exog_test)
        rmse = np.sqrt(mean_squared_error(y_sarimax_test, preds))
        if rmse < best_rmse:
            best_rmse, best_order, best_sarimax = rmse, (p, d, q), res
    except:
        continue

print(f"Best SARIMAX order: {best_order} | RMSE: {best_rmse:.2f}")

preds_sarimax_tuned = best_sarimax.forecast(steps=len(y_sarimax_test), exog=exog_test)
metrics_sarimax_tuned = evaluate(y_sarimax_test, preds_sarimax_tuned, "SARIMAX Tuned")

with mlflow.start_run(run_name="sarimax-tuned"):
    mlflow.log_params({"p": best_order[0], "d": best_order[1], "q": best_order[2]})
    mlflow.log_metrics(metrics_sarimax_tuned)


# ---
## S7 - SARIMAX Grid Search, oil removed


# SARIMAX No-Oil: Grid Search + Best Model Log
from itertools import product

EXOG_COLS_NO_OIL = ['is_national_holiday', 'is_regional_holiday', 'is_local_holiday']

exog_train_no_oil = df_train[EXOG_COLS_NO_OIL].fillna(0)
exog_test_no_oil  = df_test[EXOG_COLS_NO_OIL].fillna(0)

best_rmse_no_oil, best_order_no_oil, best_model_no_oil = float('inf'), None, None

for p, d, q in product(p_values, d_values, q_values):
    try:
        m = SARIMAX(y_sarimax_train, exog=exog_train_no_oil, order=(p, d, q),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        preds = res.forecast(steps=len(y_sarimax_test), exog=exog_test_no_oil)
        rmse = np.sqrt(mean_squared_error(y_sarimax_test, preds))
        if rmse < best_rmse_no_oil:
            best_rmse_no_oil, best_order_no_oil, best_model_no_oil = rmse, (p, d, q), res
    except:
        continue

print(f"Best no-oil order: {best_order_no_oil} | RMSE: {best_rmse_no_oil:.2f}")

preds_no_oil = best_model_no_oil.forecast(steps=len(y_sarimax_test), exog=exog_test_no_oil)

with mlflow.start_run(run_name="sarimax-no-oil"):
    metrics_no_oil = evaluate(y_sarimax_test, preds_no_oil, "SARIMAX No-Oil")
    mlflow.log_params({"p": best_order_no_oil[0], "d": best_order_no_oil[1],
                       "q": best_order_no_oil[2], "exog": "holidays-only"})
    mlflow.log_metrics(metrics_no_oil)


# ---
## S8 - Prophet Baseline & variations


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


# Prophet Tuning: Variations + Grid Search
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- Predefined Variations ---
variations = {
    "prophet-v1-multiplicative": {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 20,
        "weekly_seasonality": True,
        "yearly_seasonality": False,
        "daily_seasonality": False,
    },
    "prophet-v2-aggressive": {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.3,
        "seasonality_prior_scale": 5,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "daily_seasonality": False,
    },
}

best_prophet_rmse  = float('inf')
best_prophet_model = None
best_prophet_preds = None
best_prophet_name  = None

# Run baseline through same tracking loop
all_prophet_runs = {
    "prophet-baseline": {
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "daily_seasonality": False,
    },
    **variations
}

for run_name, params in all_prophet_runs.items():
    m = Prophet(**params)
    m.fit(prophet_train)
    future   = m.make_future_dataframe(periods=len(prophet_test))
    forecast = m.predict(future)
    preds    = forecast['yhat'].tail(len(prophet_test)).values
    actuals  = prophet_test['y'].values

    with mlflow.start_run(run_name=run_name):
        metrics_p = evaluate(actuals, preds, run_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics_p)

    if metrics_p['rmse'] < best_prophet_rmse:
        best_prophet_rmse  = metrics_p['rmse']
        best_prophet_model = m
        best_prophet_preds = preds
        best_prophet_name  = run_name

print(f"\nBest variation: {best_prophet_name} | RMSE: {best_prophet_rmse:.2f}")

# --- Grid Search ---
print("\nRunning Prophet grid search...")
param_grid = {
    'changepoint_prior_scale': [0.05, 0.1, 0.3, 0.5],
    'seasonality_prior_scale': [5, 10, 20],
    'seasonality_mode':        ['additive', 'multiplicative'],
}

grid_keys   = list(param_grid.keys())
grid_values = list(param_grid.values())

grid_best_rmse   = float('inf')
grid_best_params = None
grid_best_preds  = None

for combo in product(*grid_values):
    params = dict(zip(grid_keys, combo))
    run_name = (f"prophet-grid-"
                f"cp{params['changepoint_prior_scale']}-"
                f"sp{params['seasonality_prior_scale']}-"
                f"{params['seasonality_mode'][:3]}")
    try:
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            **params
        )
        m.fit(prophet_train)
        future   = m.make_future_dataframe(periods=len(prophet_test))
        forecast = m.predict(future)
        preds    = forecast['yhat'].tail(len(prophet_test)).values
        actuals  = prophet_test['y'].values

        with mlflow.start_run(run_name=run_name):
            metrics_g = evaluate(actuals, preds, run_name)
            mlflow.log_params({
                **params,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "daily_seasonality":  False,
                "source": "grid_search"
            })
            mlflow.log_metrics(metrics_g)

        if metrics_g['rmse'] < grid_best_rmse:
            grid_best_rmse   = metrics_g['rmse']
            grid_best_params = params
            grid_best_preds  = preds

    except Exception as e:
        print(f"Failed: {run_name} -- {e}")
        continue

print(f"\nGrid best RMSE: {grid_best_rmse:.2f}")
print(f"Grid best params: {grid_best_params}")

# --- Update preds_prophet to best overall ---
if grid_best_rmse < best_prophet_rmse:
    preds_prophet = grid_best_preds
    print(f"Champion: grid search ({grid_best_rmse:.2f})")
else:
    preds_prophet = best_prophet_preds
    print(f"Champion: {best_prophet_name} ({best_prophet_rmse:.2f})")


# ---
## S9 - Holt-Winters


# Holt-Winters (Exponential Smoothing) Baseline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

with mlflow.start_run(run_name="holtwinters-baseline"):
    model_hw = ExponentialSmoothing(
        y_sarimax_train,
        trend='add',
        seasonal='add',
        seasonal_periods=7
    ).fit()

    preds_hw = model_hw.forecast(steps=len(y_sarimax_test))
    metrics_hw = evaluate(y_sarimax_test.values, preds_hw.values, "Holt-Winters Baseline")

    mlflow.log_params({"trend": "add", "seasonal": "add", "seasonal_periods": 7})
    mlflow.log_metrics(metrics_hw)


# ---
## S8 - Save Classical & ML Model Predictions


# Save Classical & ML Model Predictions
import pandas as pd

predictions_to_save = {
    'xgboost':      (y_test.index,              y_test.values,         preds_xgb),
    'sarimax':      (y_sarimax_test.index,       y_sarimax_test.values, preds_sarimax.values),
    'prophet': (
        prophet_test['ds'].values[-len(preds_prophet):],
        prophet_test['y'].values[-len(preds_prophet):],
        preds_prophet
),
    'holtwinters':  (y_sarimax_test.index,       y_sarimax_test.values, preds_hw.values),
}

for model_name, (idx, actual, pred) in predictions_to_save.items():
    df_out = pd.DataFrame({
        'date':      idx,
        'actual':    actual,
        'predicted': pred,
        'residual':  actual - pred
    }).set_index('date')
    
    out_path = os.path.join(config.MODELS_PATH, f'{model_name}_predictions.csv')
    df_out.to_csv(out_path)
    print(f"Saved: {out_path} | rows: {len(df_out)}")


# ---
## S10 - Results Summary
# Compares all logged runs from this experiment in a single table.


runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT])

summary = runs[["tags.mlflow.runName", "metrics.rmse", "metrics.mae", 
                "metrics.mape", "metrics.r2"]].rename(
    columns={"tags.mlflow.runName": "model"}
).sort_values("metrics.rmse")

print(summary.to_string(index=False))


# ---
## S11 - Visual Comparison


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
## S12 -  Save Best Model


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
# 
## S13 - XGBoost Hyperopt Tuning - baseline


# XGBoost Hyperopt Tuning
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

xgb_space = {
    'n_estimators':      hp.choice('n_estimators', [200, 500, 1000]),
    'max_depth':         hp.choice('max_depth', [3, 5, 7]),
    'learning_rate':     hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample':         hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree':  hp.uniform('colsample_bytree', 0.6, 1.0),
}

def xgb_objective(params):
    m = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        verbosity=0
    )
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {'loss': rmse, 'status': STATUS_OK}

xgb_trials = Trials()
xgb_best = fmin(fn=xgb_objective, space=xgb_space, algo=tpe.suggest,
                max_evals=20, trials=xgb_trials)

tuned_xgb_params = {
    'n_estimators':     [200, 500, 1000][xgb_best['n_estimators']],
    'max_depth':        [3, 5, 7][xgb_best['max_depth']],
    'learning_rate':    xgb_best['learning_rate'],
    'subsample':        xgb_best['subsample'],
    'colsample_bytree': xgb_best['colsample_bytree'],
}

with mlflow.start_run(run_name="xgboost-tuned"):
    model_xgb_tuned = xgb.XGBRegressor(**tuned_xgb_params)
    model_xgb_tuned.fit(X_train, y_train)
    preds_xgb_tuned = model_xgb_tuned.predict(X_test)
    metrics_xgb_tuned = evaluate(y_test, preds_xgb_tuned, "XGBoost Tuned")
    mlflow.log_params(tuned_xgb_params)
    mlflow.log_metrics(metrics_xgb_tuned)
    mlflow.xgboost.log_model(model_xgb_tuned, "model")

print(f"\nBest params: {tuned_xgb_params}")


# ---  
# 
## S14 - XGBoost Hyperopt Tuned, No-Exogenous Features


# XGBoost Hyperopt Tuned, No-Exogenous Features
FEATURES_NO_EXO = [
    'year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_7d_mean', 'rolling_14d_mean', 'rolling_30d_mean', 'rolling_7d_std'
]

X_train_ne = df_train[FEATURES_NO_EXO].fillna(0)
X_test_ne  = df_test[FEATURES_NO_EXO].fillna(0)

def xgb_objective_ne(params):
    m = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        verbosity=0
    )
    m.fit(X_train_ne, y_train)
    preds = m.predict(X_test_ne)
    return {'loss': np.sqrt(mean_squared_error(y_test, preds)), 'status': STATUS_OK}

xgb_ne_trials = Trials()
xgb_ne_best = fmin(fn=xgb_objective_ne, space=xgb_space,
                   algo=tpe.suggest, max_evals=20, trials=xgb_ne_trials)

tuned_ne_params = {
    'n_estimators':     [200, 500, 1000][xgb_ne_best['n_estimators']],
    'max_depth':        [3, 5, 7][xgb_ne_best['max_depth']],
    'learning_rate':    xgb_ne_best['learning_rate'],
    'subsample':        xgb_ne_best['subsample'],
    'colsample_bytree': xgb_ne_best['colsample_bytree'],
}

with mlflow.start_run(run_name="xgb-no-exo-tuned"):
    m_ne = xgb.XGBRegressor(**tuned_ne_params)
    m_ne.fit(X_train_ne, y_train)
    preds_ne = m_ne.predict(X_test_ne)
    metrics_ne = evaluate(y_test, preds_ne, "XGBoost No-Exo Tuned")
    mlflow.log_params({**tuned_ne_params, "features": "no-exogenous"})
    mlflow.log_metrics(metrics_ne)


# ---
## S15 - Notes & Observations
# Document findings per experiment run.
# 
# - Best model this run: XGBoost
# - Notable differences between models:
# - Features that helped/hurt:
# - Next experiment to try:
# - Next steps with this Notebook: Redefine feature ablation variable considerations with more templated approach.
#     - ex: Change 'no-oil' labels and other dataset specific labels and variables to something more applicable to any dataset


