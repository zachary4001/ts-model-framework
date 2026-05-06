# ── app.py — Time Series Model Comparison Dashboard ──────────
# Version 1.07
# updated 06.05.2026
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
# if 'auto_run_done' not in st.session_state:
#     st.session_state.auto_run_done = True
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
st.caption("Retail Unit Sales — Corporación Favorita dataset")

# ── Additional session state ──────────────────────────────────
for _key, _default in [
    ('auto_run_done', None),
    ('last_model',    None),
    ('last_n_days',   None),
    ('last_cutoff',   None),
    ('forecast_df',   None),
    ('theme',         'light'),
    ('show_about',    False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Eager first-load forecast ─────────────────────────────────
# Runs before any UI renders so forecast_df is populated when the
# sidebar download button is evaluated for the first time.
if not st.session_state.auto_run_done:
    _default_cutoff = pd.to_datetime("2013-12-31")
    _default_n_days = 30
    _df0 = load_data()
    _m0, _sc0, _p0 = load_torch_model('rnn')
    if _m0 is not None:
        st.session_state.forecast_df   = make_torch_forecast(_df0, _m0, _sc0, _p0, _default_cutoff, _default_n_days)
        st.session_state.auto_run_done = True
        st.session_state.last_model    = "RNN (tuned)"
        st.session_state.last_n_days   = _default_n_days
        st.session_state.last_cutoff   = str(_default_cutoff.date())

# ── Theme CSS ─────────────────────────────────────────────────
_THEMES = {
    'light': {
        'bg':      '#f0f4f8',
        'sidebar': '#dce8f5',
        'widget':  '#ffffff',
        'text':    '#1a1a2e',
        'accent':  '#2962ff',
        'border':  'rgba(0,0,0,0.15)',
    },
    'dark': {
        'bg':      '#161b22',
        'sidebar': '#161b22',
        'widget':  '#1c2733',
        'text':    '#e6edf3',
        'accent':  '#4fc3f7',
        'border':  'rgba(255,255,255,0.15)',
    },
}
_t = _THEMES[st.session_state.theme]
_dark_extra = """
    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #4fc3f7;
        color: #4fc3f7;
    }
    /* Expander titles */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #e6edf3 !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #21262d;
    }
    /* Expander body */
    .streamlit-expanderContent {
        background-color: #0d1117;
        border-color: #30363d;
    }
    /* Download button */
    .stDownloadButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
    .stDownloadButton > button:hover {
        border-color: #4fc3f7;
        color: #4fc3f7;
    }
    /* Expander header -- target underlying details/summary element */
    details > summary {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    details > summary:hover {
        background-color: #21262d !important;
    }
    details {
        border-color: #30363d !important;
    }
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom-color: #4fc3f7 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e6edf3 !important;
    }
    /* Selectbox labels in main content */
    .stSelectbox label, .stDateInput label, .stSlider label {
        color: #e6edf3 !important;
    }
    /* Metric tiles -- label, value, and help icon */
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
    }
    [data-testid="stMetricLabel"] svg {
        fill: #8b949e !important;
    }
    /* Subheaders and headings in main content */
    h1, h2, h3, h4 {
        color: #e6edf3 !important;
    }
    /* st.info / st.warning text */
    [data-testid="stAlert"] p {
        color: #e6edf3 !important;
    }
    /* st.Popover size */
    .stPopover [data-testid="stPopoverBody"] 
    { width: 100px !important;
    max-width: 100px !important; }
    { min-width: 130px; } a
"""
st.markdown(f"""
<style>
    .stApp {{
        background-color: {_t['bg']};
        color: {_t['text']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {_t['sidebar']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {_t['text']};
    }}
    /* Selectbox / dropdown widgets */
    div[data-baseweb="select"] > div {{
        background-color: {_t['widget']} !important;
        border: 1px solid {_t['border']} !important;
        color: {_t['text']} !important;
    }}
    .stSelectbox > div > div {{
        background-color: {_t['widget']} !important;
        border: 1px solid {_t['border']} !important;
    }}
    /* Date input */
    div[data-testid="stDateInput"] input {{
        background-color: {_t['widget']} !important;
        border: 1px solid {_t['border']} !important;
        color: {_t['text']} !important;
    }}
    /* Slider track area */
    div[data-testid="stSlider"] {{
        background-color: transparent;
    }}
    div[data-testid="stSlider"] > div {{
        border: 1px solid {_t['border']};
        border-radius: 6px;
        padding: 4px 8px;
        background-color: {_t['widget']};
    }}
    /* Hide Streamlit header */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    /* Centre theme-toggle button row in sidebar */
    section[data-testid="stSidebar"] .stHorizontalBlock {{
        margin: 0 auto;
        width: fit-content;
    }}
    {_dark_extra if st.session_state.theme == 'dark' else ''}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────

# Theme toggle + about — top row
st.sidebar.markdown(
    '<div style="display:flex; justify-content:center; margin-bottom:8px;">',
    unsafe_allow_html=True
)
_tcol, _acol, _ = st.sidebar.columns([3.5, 3, 3])
if _tcol.button("🌓", help="Toggle light/dark theme", use_container_width=False):
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

_st_version = tuple(int(x) for x in st.__version__.split('.')[:2])
_has_popover = _st_version >= (1, 31)

_ABOUT_TEXT = """
**Corporación Favorita — Retail Sales Forecaster**
Store 44 | Item 1047679

This tool forecasts daily unit sales using multiple time-series models
trained on 2013 historical data and validated on Q1 2014.

**Models available:** RNN, LSTM, XGBoost, Prophet, Holt-Winters
**Champion model:** RNN (tuned) | RMSE: 104.46
**Built with:** Python, PyTorch, Streamlit, MLflow

*[📄 Full README and documentation on GitHub](https://github.com/zachary4001/ts-model-framework/blob/main/README.md)*
"""

def _render_popover_content():
    st.markdown("**About**")
    st.markdown(_ABOUT_TEXT)
    #st.divider()
    st.markdown("<hr style='margin:6px 0; border-color:#30363d'>", unsafe_allow_html=True)
    st.markdown("**Theme**")
    if st.button("Toggle light/dark", key="theme_popover", use_container_width=False):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    st.divider()
    st.markdown("**Download**")
    if st.session_state.forecast_df is not None:
        st.download_button(
            label="Download forecast CSV",
            data=st.session_state.forecast_df.reset_index().to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv",
            use_container_width=False,
            key="dl_popover",
        )
    else:
        st.download_button(
            label="Download forecast CSV",
            data="", file_name="forecast.csv", mime="text/csv",
            disabled=True, use_container_width=False, key="dl_popover_disabled",
        )
        st.caption("Run a forecast to enable download.")

if _has_popover:
    with _acol.popover("⚙", help="Settings & info", use_container_width=False):
        _render_popover_content()
else:
    if _acol.button("⚙", help="Settings & info", use_container_width=False):
        st.session_state.show_about = not st.session_state.show_about
    if st.session_state.show_about:
        with st.sidebar.expander("Settings & info", expanded=True):
            _render_popover_content()

st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.header("Forecast Settings")
st.sidebar.caption("Data range: 2013-01-02 → 2014-03-31")

# ── Store selector ────────────────────────────────────────────
@st.cache_data
def load_stores():
    df = pd.read_csv(os.path.join(config.DATA_PATH, 'stores.csv'))
    df = df.sort_values('store_nbr')
    df['label'] = df.apply(
        lambda r: f"Store {r['store_nbr']}, {r['city']}, {r['region']}", axis=1
    )
    return df

_stores = load_stores()
_default_store_idx = int(_stores[_stores['store_nbr'] == 44].index[0]) if 44 in _stores['store_nbr'].values else 0
_default_store_idx = _stores.index.get_loc(_default_store_idx)
store_selector = st.sidebar.selectbox(
    "Store",
    options=_stores['label'].tolist(),
    index=_default_store_idx,
)

# ── Product category selector ─────────────────────────────────
CATEGORY_OPTIONS = [
    'Total Unit Sales', 'Produce', 'Dry Goods', 'Bread & Pastries',
    'Dairy', 'Beverages', 'Snacks & Candy', 'Cleaning Products',
    'Personal Care', 'Frozen Foods', 'Meats & Seafood',
    'Canned Goods', 'Baby Products',
]
category_selector = st.sidebar.selectbox("Product Category", CATEGORY_OPTIONS, index=0)

MODEL_OPTIONS = ["RNN (tuned)", "LSTM (tuned)", "XGBoost", "Prophet", "Holt-Winters"]
model_selector = st.sidebar.selectbox("Forecast model", MODEL_OPTIONS, index=0)

cutoff_date = st.sidebar.date_input(
    "Cutoff Date",
    value=pd.to_datetime("2013-12-31"),
    min_value=pd.to_datetime("2013-06-01"),
    max_value=pd.to_datetime("2014-03-30")
)
n_days       = st.sidebar.slider("Days to forecast", 1, 90, 30)
if n_days > 30:
    st.sidebar.info("⚠ Forecasts beyond 30 days have reduced reliability for this dataset")
history_days = st.sidebar.slider("History days to show", 14, 120, 90)

# Download button — always visible, disabled until forecast exists
st.sidebar.divider()
_has_forecast = st.session_state.forecast_df is not None
if _has_forecast:
    _csv = st.session_state.forecast_df.reset_index().to_csv(index=False)
    st.sidebar.download_button(
        label="Download forecast CSV",
        data=_csv,
        file_name=f"forecast_{cutoff_date}_{model_selector}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.sidebar.download_button(
        label="Download forecast CSV",
        data="",
        file_name="forecast.csv",
        mime="text/csv",
        disabled=True,
        use_container_width=True,
    )
    st.sidebar.caption("Run a forecast to enable download.")

model_display_name = load_model_name() if model_selector == "XGBoost" else model_selector
st.info(f"Active model: **{model_display_name}**")

# ── Constants ─────────────────────────────────────────────────
PRED_CSV_MAP = {
    "RNN (tuned)":  "rnn_predictions.csv",
    "LSTM (tuned)": "lstm_predictions.csv",
    "XGBoost":      "xgboost_predictions.csv",
    "Prophet":      "prophet_predictions.csv",
    "Holt-Winters": "holtwinters_predictions.csv",
}

FAMILY_MAP = {
    'rnn': 'RNN', 'lstm': 'LSTM', 'xgboost': 'XGBoost',
    'prophet': 'Prophet', 'holtwinters': 'Holt-Winters', 'sarimax': 'SARIMAX'
}

# ── Helpers ───────────────────────────────────────────────────
def load_classical_model(selector):
    file_map = {
        "XGBoost":      "best_model.pkl",
        "Prophet":      "prophet_model.pkl",
        "Holt-Winters": "holtwinters_model.pkl",
    }
    fname = file_map.get(selector)
    if fname:
        path = os.path.join(config.MODELS_PATH, fname)
        if os.path.exists(path):
            return joblib.load(path)
    return None


def get_model_metrics(selector):
    selector_to_keyword = {
        "RNN (tuned)":  "rnn",
        "LSTM (tuned)": "lstm",
        "XGBoost":      "xgboost",
        "Prophet":      "prophet",
        "Holt-Winters": "holtwinters",
    }
    results = load_mlflow_results()
    if results is None or "model" not in results.columns:
        return None
    keyword = selector_to_keyword.get(selector, "").lower()
    mask = results["model"].str.lower().str.contains(keyword, na=False)
    filtered = results[mask]
    if filtered.empty:
        return None
    best = filtered.sort_values("RMSE").iloc[0]
    return {k: best.get(k) for k in ["RMSE", "MAE", "Bias"]}


def get_family(name):
    name_lower = str(name).lower()
    for key, label in FAMILY_MAP.items():
        if key in name_lower:
            return label
    return "Other"


def get_tuning(name):
    name_lower = str(name).lower()
    if 'baseline' in name_lower:
        return 'Baseline'
    elif 'tuned' in name_lower:
        return 'Hyperopt Tuned'
    elif 'grid' in name_lower:
        return 'Grid Search'
    elif 'v1' in name_lower or 'v2' in name_lower:
        return 'Variation'
    return 'Other'


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Forecast", "Model Comparison", "Data Insights"])

# ────────────────────────────────────────────────────────────
# Tab 1 — Forecast
# ────────────────────────────────────────────────────────────
with tab1:
    cutoff = pd.to_datetime(cutoff_date)

    # Re-run forecast when model, n_days, or cutoff changes (not history_days —
    # that only affects how much history is shown, no model call needed)
    model_changed  = st.session_state.last_model  != model_selector
    ndays_changed  = st.session_state.last_n_days  != n_days
    cutoff_changed = st.session_state.last_cutoff  != str(cutoff_date)
    should_run     = not st.session_state.auto_run_done or model_changed or ndays_changed or cutoff_changed

    if should_run:
        st.session_state.auto_run_done = False  # prevent infinite rerun loop
        # existing forecast generation code unchanged
        df = load_data()

        with st.spinner(f"Generating {model_selector} forecast..."):
            if model_selector in ("RNN (tuned)", "LSTM (tuned)"):
                mt = 'rnn' if model_selector == "RNN (tuned)" else 'lstm'
                model, scaler, params = load_torch_model(mt)
                if model is None:
                    st.error(f"{model_selector} model not found. Run the corresponding notebook first.")
                    st.stop()
                forecast_df = make_torch_forecast(df, model, scaler, params, cutoff, n_days)

            else:
                model = load_classical_model(model_selector)
                if model is None:
                    model = load_model()
                if model is None:
                    st.error(f"No saved model found for {model_selector}.")
                    st.stop()
                forecast_df = make_forecast(df, model, FEATURES, cutoff, n_days)

        st.session_state.forecast_df   = forecast_df
        st.session_state.auto_run_done = True
        st.session_state.last_model    = model_selector
        st.session_state.last_n_days   = n_days
        st.session_state.last_cutoff   = str(cutoff_date)

    if st.session_state.forecast_df is not None:
        df           = load_data()  # cached, free to call
        forecast_df  = st.session_state.forecast_df

        # Always recompute history_plot from current history_days slider
        history_plot = df.loc[
            (df.index >= cutoff - pd.Timedelta(days=history_days)) &
            (df.index <= cutoff)
        ][config.TARGET_COLUMN]

        chart_view = st.selectbox("View", ["Forecast View", "Actual vs Predicted"])

        # ── Forecast View ─────────────────────────────────────
        if chart_view == "Forecast View":
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
                title=f"Sales Forecast from {cutoff.date()} — {model_selector}",
                yaxis_title="Unit Sales",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Actual vs Predicted ───────────────────────────────
        else:
            csv_name  = PRED_CSV_MAP.get(model_selector)
            pred_path = os.path.join(config.MODELS_PATH, csv_name) if csv_name else None

            if pred_path and os.path.exists(pred_path):
                pred_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
                fig = go.Figure()
                if "actual" in pred_df.columns:
                    fig.add_trace(go.Scatter(
                        x=pred_df.index, y=pred_df["actual"],
                        name="Actual", line=dict(color="steelblue", width=2)
                    ))
                pred_col = next((c for c in pred_df.columns if c != "actual"), None)
                if pred_col:
                    fig.add_trace(go.Scatter(
                        x=pred_df.index, y=pred_df[pred_col],
                        name="Predicted", line=dict(color="orange", width=2, dash="dash")
                    ))
                fig.update_layout(
                    title=f"Actual vs Predicted — {model_selector}",
                    yaxis_title="Unit Sales",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    f"Predictions CSV not found for **{model_selector}**. "
                    "Run the corresponding model notebook first."
                )

        # ── Metric tiles ──────────────────────────────────────
        metrics = get_model_metrics(model_selector)
        m1, m2, m3 = st.columns(3)

        def fmt(val):
            return f"{val:.2f}" if val is not None and not pd.isna(val) else "N/A"

        m1.metric(
            "RMSE", fmt(metrics.get("RMSE") if metrics else None),
            help="Root Mean Squared Error — penalizes large misses more heavily"
        )
        m2.metric(
            "MAE", fmt(metrics.get("MAE") if metrics else None),
            help="Mean Absolute Error — average units off per day"
        )
        m3.metric(
            "Bias", fmt(metrics.get("Bias") if metrics else None),
            help="Systematic over/under prediction — negative means model tends to under-predict"
        )

        # ── Forecast table ────────────────────────────────────
        st.subheader("Forecast values")
        st.dataframe(forecast_df.reset_index().rename(
            columns={"date": "Date", "forecast": "Predicted Sales"}
        ))

# ────────────────────────────────────────────────────────────
# Tab 2 — Model Comparison
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Comparison")

    with st.expander("📖 Metric Definitions"):
        st.markdown("""
- **RMSE** — Root Mean Squared Error. Penalizes large misses heavily. Lower is better.
- **MAE** — Mean Absolute Error. Average miss per day in units. Lower is better.
- **MAPE** — Mean Absolute Percentage Error. Average miss as % of actual. Lower is better.
- **R²** — How much of the sales pattern the model explains. Closer to 1.0 is better.
- **Bias** — Systematic over or under-prediction. Closer to 0 is better. Negative = under-predicts.
""")

    results = load_mlflow_results()

    if results is not None:
        results = results.copy()
        results["Family"] = results["model"].apply(get_family)
        results["Tuning"] = results["model"].apply(get_tuning)

        # Best RMSE per family
        best = (
            results.sort_values("RMSE")
            .groupby("Family", sort=False)
            .first()
            .reset_index()
            .sort_values("RMSE")
        )
        display_cols = [c for c in ["Family", "Tuning", "RMSE", "MAE", "Bias", "R²", "MAPE"]
                        if c in best.columns]
        _display = best[display_cols].reset_index(drop=True)
        _win_color = '#d4edda' if st.session_state.theme == 'light' else '#d4edda'

        def _highlight_best(row):
            return [
                f'background-color: {_win_color}' if row.name == _display['RMSE'].idxmin() else ''
                for _ in row
            ]

        st.dataframe(_display.style.apply(_highlight_best, axis=1))

        fig2 = go.Figure(go.Bar(
            x=best["RMSE"],
            y=best["Family"],
            orientation='h',
            marker_color="steelblue",
            text=best["RMSE"].round(2),
            textposition="outside"
        ))
        fig2.update_layout(
            title="RMSE by Model Family (lower is better)",
            xaxis_title="RMSE",
            height=350,
            margin=dict(l=130)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning(
            "No MLflow results found. "
            "Make sure the MLflow server is running on localhost:5000."
        )

# ────────────────────────────────────────────────────────────
# Tab 3 — Data Insights
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Data Insights")

    IMAGE_INFO = [
        ("sales_by_dayofweek.png",     "Sales by Day of Week"),
        ("xgb_feature_importance-1.png", "XGBoost Feature Importance"),
        ("oil_sales_rescaled.png",     "Oil Price vs Sales (Rescaled)"),
        ("residuals_over_time.png",    "Residuals Over Time"),
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)
    grid = [row1[0], row1[1], row2[0], row2[1]]

    for cell, (fname, caption) in zip(grid, IMAGE_INFO):
        with cell:
            with st.expander(caption, expanded=True):
                img_path = os.path.join(config.MODELS_PATH, fname)
                if os.path.exists(img_path):
                    st.image(img_path, caption=caption, use_container_width=True)
                else:
                    st.info(f"Image not found: {fname}")
