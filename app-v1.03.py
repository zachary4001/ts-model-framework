# ── app.py — Time Series Model Comparison Dashboard ──────────
# Version 1.03
# updated 04.05.2026
# ────────────────────────────────────────────────────────────
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sys
sys.path.append(r'Q:\\scripts\\projects\\ts-model-framework')
import config

# DATA_PATH   = r"Q:\scripts\projects\ts-model-framework\data"
# MODELS_PATH = r"Q:\scripts\projects\ts-model-framework\models"

# DATE_COLUMN   = "date"
# TARGET_COLUMN = "unit_sales"
FEATURES = [
    'year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_7d_mean', 'rolling_14d_mean', 'rolling_30d_mean', 'rolling_7d_std',
    'dcoilwtico', 'oil_lag_1', 'oil_rolling_7d_mean',
    'is_national_holiday', 'is_regional_holiday', 'is_local_holiday',
]

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(config.DATA_PATH, 'timeseries_with_features.csv'))
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).set_index(config.DATE_COLUMN)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range).fillna(0)
    return df

# ── Load best model ───────────────────────────────────────────
# @st.cache_resource
def load_model():
    path = os.path.join(config.MODELS_PATH, 'best_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_model_name():
    path = os.path.join(config.MODELS_PATH, 'best_model_name.txt')
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "Unknown"

# ── Forecast function ─────────────────────────────────────────
def make_forecast(df, model, features, cutoff, n_days):
    cutoff     = pd.to_datetime(cutoff)
    model_type = type(model).__name__

    if 'SARIMAXResults' in model_type or 'ARIMAResults' in model_type:
        preds = model.forecast(steps=n_days)
        return pd.DataFrame({
            'date':     [cutoff + pd.Timedelta(days=i+1) for i in range(n_days)],
            'forecast': [round(float(max(0, p)), 2) for p in preds]
        }).set_index('date')

    elif 'HoltWintersResults' in model_type or 'ExponentialSmoothing' in model_type:
        preds = model.forecast(n_days)
        return pd.DataFrame({
            'date':     [cutoff + pd.Timedelta(days=i+1) for i in range(n_days)],
            'forecast': [round(float(max(0, p)), 2) for p in preds]
        }).set_index('date')

    elif 'Prophet' in model_type:
        future_dates = pd.date_range(
            start=cutoff + pd.Timedelta(days=1), periods=n_days, freq='D'
        )
        forecast_out = model.predict(pd.DataFrame({'ds': future_dates}))
        return pd.DataFrame({
            'date':     forecast_out['ds'].values,
            'forecast': [round(float(max(0, y)), 2) for y in forecast_out['yhat']]
        }).set_index('date')

    else:
        history   = df.loc[df.index <= cutoff].copy()
        forecasts = []
        for i in range(n_days):
            next_date = cutoff + pd.Timedelta(days=i+1)
            row = df.loc[[next_date], features] if next_date in df.index \
                  else history.iloc[[-1]][features].copy()
            row.index = [next_date]
            forecasts.append({
                'date':     next_date,
                'forecast': round(float(max(0, model.predict(row)[0])), 2)
            })
        return pd.DataFrame(forecasts).set_index('date')

# ── MLflow leaderboard ────────────────────────────────────────
@st.cache_data
def load_mlflow_results():
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT])
        if runs.empty:
            return None
        cols = {
            "tags.mlflow.runName": "model",
            "metrics.rmse": "RMSE",
            "metrics.mae": "MAE",
            "metrics.mape": "MAPE",
            "metrics.r2": "R²",
            "metrics.bias": "Bias"
        }
        available = {k: v for k, v in cols.items() if k in runs.columns}
        summary = runs[list(available.keys())].rename(columns=available)
        summary = summary.dropna(subset=["RMSE"]).sort_values("RMSE")
        return summary
    except Exception:
        return None

# ── App layout ────────────────────────────────────────────────
st.set_page_config(page_title="TS Model Framework", layout="wide")
st.title("Time-Series Model Comparison")
st.caption("Retail Unit Sales - Corporacion Favorita dataset")

model_name = load_model_name()
st.info(f"Active model: **{model_name}**")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Forecast Settings")

cutoff_date  = st.sidebar.date_input(
    "Cutoff date",
    value=pd.to_datetime("2014-01-15"),
    min_value=pd.to_datetime("2013-06-01"),
    max_value=pd.to_datetime("2014-03-30")
)
n_days       = st.sidebar.slider("Days to forecast", 1, 30, 7)
history_days = st.sidebar.slider("History days to show", 14, 120, 60)
run_button   = st.sidebar.button("Run Forecast")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Forecast", "Model Leaderboard"])

with tab1:
    if run_button:
        model = load_model()
        if model is None:
            st.error("No model found. Run 02_experiments.ipynb first.")
        else:
            df = load_data()
            cutoff = pd.to_datetime(cutoff_date)

            history_plot = df.loc[
                (df.index >= cutoff - pd.Timedelta(days=history_days)) &
                (df.index <= cutoff)
            ][config.TARGET_COLUMN]

            with st.spinner("Generating forecast..."):
                forecast_df = make_forecast(df, model, FEATURES, cutoff, n_days)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(history_plot.index, history_plot.values,
                    label="Historical sales", color="steelblue", linewidth=1.5)
            ax.plot(forecast_df.index, forecast_df["forecast"].values,
                    label=f"{n_days}-day forecast", color="orange",
                    linestyle="--", linewidth=2, marker="o", markersize=4)
            ax.axvline(cutoff, color="red", linestyle=":", linewidth=1.5,
                       label="Cutoff date")
            ax.set_title(f"Sales Forecast from {cutoff.date()}")
            ax.set_ylabel("Unit Sales")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Forecast values")
            st.dataframe(forecast_df.reset_index().rename(
                columns={"date": "Date", "forecast": "Predicted Sales"}
            ))

            csv = forecast_df.reset_index().to_csv(index=False)
            st.download_button(
                label="Download forecast as CSV",
                data=csv,
                file_name=f"forecast_{cutoff_date}.csv",
                mime="text/csv"
            )
            st.success("Forecast complete!")
    else:
        st.info("Adjust settings in the sidebar and click **Run Forecast**.")

with tab2:
    st.subheader("MLflow Model Leaderboard")
    results = load_mlflow_results()
    if results is not None:
        st.dataframe(results.reset_index(drop=True))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(results["model"], results["RMSE"], color="steelblue")
        ax2.set_title("RMSE by Model (lower is better)")
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("No MLflow results found. Make sure the MLflow server is running on localhost:5000.")