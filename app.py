# ── app.py — Time Series Model Comparison Dashboard ──────────
# Version 1.04
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
import plotly.graph_objects as go

if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'history_plot' not in st.session_state:
    st.session_state.history_plot = None

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

# ── Load LSTM / RNN models ────────────────────────────────────
def load_torch_model(model_type='lstm'):
    import torch, json
    from torch import nn

    params_path = os.path.join(config.MODELS_PATH, f'{model_type}_params.json')
    weights_path = os.path.join(config.MODELS_PATH, f'best_{model_type}_model.pt')
    scaler_path = os.path.join(config.MODELS_PATH, f'{model_type}_scaler.pkl')

    if not all(os.path.exists(p) for p in [params_path, weights_path, scaler_path]):
        return None, None, None

    with open(params_path) as f:
        params = json.load(f)

    class LSTMForecaster(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class SimpleRNN(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, output_size=1):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    hidden_size = params.get('hidden_size', 64)
    ModelClass = LSTMForecaster if model_type == 'lstm' else SimpleRNN
    model = ModelClass(hidden_size=hidden_size)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler, params


def make_torch_forecast(df, model, scaler, params, cutoff, n_days):
    import torch
    cutoff = pd.to_datetime(cutoff)
    seq_len = params.get('sequence_length', 30)
    target = config.TARGET_COLUMN

    history = df.loc[df.index <= cutoff][target].fillna(0).values[-seq_len:]
    if len(history) < seq_len:
        return None

    scaled = scaler.transform(history.reshape(-1, 1))
    forecasts = []

    for i in range(n_days):
        seq = torch.FloatTensor(scaled[-seq_len:]).reshape(1, seq_len, 1)
        with torch.no_grad():
            pred_scaled = model(seq).item()
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        pred = max(0, round(pred, 2))
        forecasts.append({
            'date': cutoff + pd.Timedelta(days=i+1),
            'forecast': pred
        })
        scaled = np.append(scaled, [[pred_scaled]], axis=0)

    return pd.DataFrame(forecasts).set_index('date')



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

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Forecast Settings")

cutoff_date  = st.sidebar.date_input(
    "Cutoff date",
    value=pd.to_datetime("2014-01-15"),
    min_value=pd.to_datetime("2013-06-01"),
    max_value=pd.to_datetime("2014-03-30")
)
n_days       = st.sidebar.slider("Days to forecast", 1, 30, 30)
history_days = st.sidebar.slider("History days to show", 14, 120, 60)
model_selector = st.sidebar.selectbox(
    "Forecast model",
    ["Best Classical/XGBoost", "LSTM (tuned)", "RNN (tuned)"]
)
run_button = st.sidebar.button("Run Forecast")
model_name = load_model_name() if model_selector == "Best Classical/XGBoost" else model_selector
st.info(f"Active model: **{model_name}**")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Forecast", "Model Leaderboard"])

with tab1:
    if run_button:
        df = load_data()
        cutoff = pd.to_datetime(cutoff_date)

        history_plot = df.loc[
            (df.index >= cutoff - pd.Timedelta(days=history_days)) &
            (df.index <= cutoff)
        ][config.TARGET_COLUMN]

        with st.spinner("Generating forecast..."):
            if model_selector == "Best Classical/XGBoost":
                model = load_model()
                if model is None:
                    st.error("No model found. Run 02_experiments.ipynb first.")
                    st.stop()
                forecast_df = make_forecast(df, model, FEATURES, cutoff, n_days)

            elif model_selector == "LSTM (tuned)":
                model, scaler, params = load_torch_model('lstm')
                if model is None:
                    st.error("LSTM model not found. Run 03_LSTM.ipynb first.")
                    st.stop()
                forecast_df = make_torch_forecast(df, model, scaler, params, cutoff, n_days)

            elif model_selector == "RNN (tuned)":
                model, scaler, params = load_torch_model('rnn')
                if model is None:
                    st.error("RNN model not found. Run 04_RNN.ipynb first.")
                    st.stop()
                forecast_df = make_torch_forecast(df, model, scaler, params, cutoff, n_days)

        st.session_state.forecast_df = forecast_df
        st.session_state.history_plot = history_plot

    if st.session_state.forecast_df is not None:
        cutoff = pd.to_datetime(cutoff_date)
        chart_mode = st.radio("Chart style", ["Static", "Interactive"], horizontal=True)
        forecast_df   = st.session_state.forecast_df
        history_plot  = st.session_state.history_plot

        if chart_mode == "Static":
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(history_plot.index, history_plot.values,
                    label="Historical sales", color="steelblue", linewidth=1.5)
            ax.plot(forecast_df.index, forecast_df["forecast"].values,
                    label=f"{model_selector} -- {n_days}-day forecast",
                    color="orange", linestyle="--", linewidth=2,
                    marker="o", markersize=4)
            ax.axvline(cutoff, color="red", linestyle=":", linewidth=1.5,
                    label="Cutoff date")
            ax.set_title(f"Sales Forecast from {cutoff.date()} -- {model_selector}")
            ax.set_ylabel("Unit Sales")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_plot.index, y=history_plot.values,
                name="Historical sales", line=dict(color="steelblue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df["forecast"].values,
                name=f"{model_selector} forecast",
                line=dict(color="orange", width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=6)
            ))
            fig.add_vline(
                x=cutoff.timestamp() * 1000,
                line=dict(color="red", dash="dot", width=1.5),
                annotation_text="Cutoff"
            )
            fig.update_layout(
                title=f"Sales Forecast from {cutoff.date()} -- {model_selector}",
                yaxis_title="Unit Sales",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast values")
        st.dataframe(forecast_df.reset_index().rename(
            columns={"date": "Date", "forecast": "Predicted Sales"}
        ))

        csv = forecast_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download forecast as CSV",
            data=csv,
            file_name=f"forecast_{cutoff_date}_{model_selector}.csv",
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