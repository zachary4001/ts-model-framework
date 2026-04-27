
# # 01 - Preprocessing
# Prepares the core dataset into model-ready variants.
# Each model has different data format requirements -- all handled here.
# 
# | Model | Key Requirement |
# |-------|----------------|
# | XGBoost | Engineered features, no NaN |
# | SARIMAX | Clean target series + exogenous columns |
# | Prophet | Columns renamed to `ds` and `y` |
# | LSTM/RNN | Scaled, sequenced arrays |


# ## Imports

# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH')
DATA_PATH = os.getenv('DATA_PATH')
EXPORTS_PATH = os.getenv('EXPORTS_PATH')

print(f"Project path: {PROJECT_PATH}")
print(f"Data path: {DATA_PATH}")
print(f"Data path: {EXPORTS_PATH}")
print("Libraries loaded.")


# ---
# ## S1 - Load Core Dataset
# Single source of truth -- all model variants derived from this.

# %%
FILE_NAME = "timeseries_with_features.csv"
DATE_COLUMN = "date"
TARGET_COLUMN = "unit_sales"

file_path = os.path.join(DATA_PATH, FILE_NAME)
df_core = pd.read_csv(file_path, parse_dates=[DATE_COLUMN], index_col=DATE_COLUMN)
df_core = df_core.sort_index()

print(f"Core dataset loaded: {df_core.shape}")
print(f"Date range: {df_core.index.min()} to {df_core.index.max()}")


# ---
# ## S2 - Train / Test Split
# Shared split used across ALL models for fair comparison.
# Adjust TEST_START to match your dataset.

# %%
TEST_START = "2014-01-01"   # adjust as needed

df_train = df_core[df_core.index < TEST_START].copy()
df_test  = df_core[df_core.index >= TEST_START].copy()

print(f"Train: {df_train.shape} | {df_train.index.min()} to {df_train.index.max()}")
print(f"Test:  {df_test.shape}  | {df_test.index.min()} to {df_test.index.max()}")


# ---
# ## S3 - XGBoost Variant
# Requires: engineered features, no NaN rows.

# %%
FEATURES = [
    'year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_7d_mean', 'rolling_14d_mean', 'rolling_30d_mean', 'rolling_7d_std',
    'dcoilwtico', 'oil_lag_1', 'oil_rolling_7d_mean',
    'is_national_holiday', 'is_regional_holiday', 'is_local_holiday',
]

# Drop rows where any feature is NaN
xgb_train = df_train[FEATURES + [TARGET_COLUMN]].dropna()
xgb_test  = df_test[FEATURES + [TARGET_COLUMN]].dropna()

X_train = xgb_train[FEATURES]
y_train = xgb_train[TARGET_COLUMN]
X_test  = xgb_test[FEATURES]
y_test  = xgb_test[TARGET_COLUMN]

print(f"XGBoost train: {X_train.shape} | test: {X_test.shape}")


# ---
# ## S4 - SARIMAX Variant
# Requires: clean target series + optional exogenous columns.
# NaN values forward-filled.

# %%
EXOG_COLUMNS = ['dcoilwtico', 'is_national_holiday', 'is_regional_holiday', 'is_local_holiday']

sarimax_train = df_train[[TARGET_COLUMN] + EXOG_COLUMNS].fillna(method='ffill')
sarimax_test  = df_test[[TARGET_COLUMN] + EXOG_COLUMNS].fillna(method='ffill')

y_sarimax_train = sarimax_train[TARGET_COLUMN]
y_sarimax_test  = sarimax_test[TARGET_COLUMN]
exog_train      = sarimax_train[EXOG_COLUMNS]
exog_test       = sarimax_test[EXOG_COLUMNS]

print(f"SARIMAX train: {y_sarimax_train.shape} | test: {y_sarimax_test.shape}")
print(f"Exog columns: {EXOG_COLUMNS}")


# ---
# ## S5 - Prophet Variant
# Requires: DataFrame with exactly two columns named `ds` (date) and `y` (target).

# %%
prophet_train = df_train[[TARGET_COLUMN]].reset_index().rename(
    columns={DATE_COLUMN: 'ds', TARGET_COLUMN: 'y'}
)
prophet_test = df_test[[TARGET_COLUMN]].reset_index().rename(
    columns={DATE_COLUMN: 'ds', TARGET_COLUMN: 'y'}
)

print(f"Prophet train: {prophet_train.shape} | test: {prophet_test.shape}")
print(prophet_train.head(3))


# ---
# ## S6 - LSTM/RNN Variant
# Requires: scaled values, reshaped into sequences.
# SEQUENCE_LENGTH = how many past timesteps the model sees at once.

# %%
SEQUENCE_LENGTH = 30   # adjust as needed

scaler = MinMaxScaler()

# Scale on train only -- apply same scaler to test (prevent data leakage)
train_scaled = scaler.fit_transform(df_train[[TARGET_COLUMN]].fillna(method='ffill'))
test_scaled  = scaler.transform(df_test[[TARGET_COLUMN]].fillna(method='ffill'))

def make_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_lstm_train, y_lstm_train = make_sequences(train_scaled, SEQUENCE_LENGTH)
X_lstm_test,  y_lstm_test  = make_sequences(test_scaled,  SEQUENCE_LENGTH)

# Reshape for Keras: (samples, timesteps, features)
X_lstm_train = X_lstm_train.reshape(X_lstm_train.shape[0], X_lstm_train.shape[1], 1)
X_lstm_test  = X_lstm_test.reshape(X_lstm_test.shape[0],  X_lstm_test.shape[1],  1)

print(f"LSTM train shape: {X_lstm_train.shape} | test shape: {X_lstm_test.shape}")
print(f"Scaler range: {scaler.data_min_[0]:.2f} to {scaler.data_max_[0]:.2f}")


# ---
# ## S7 - Notes & Dataset Observations
# Document any preprocessing decisions made for this specific dataset.
# 
# - Fill method chosen:
# - Exog columns selected and why:
# - SEQUENCE_LENGTH rationale:
# - Features included/excluded and why:


