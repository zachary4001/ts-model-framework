# 00_EDA
# 00 Exploratory Data Analysis  
# Reusable EDA notebook for any time series dataset.  
# Load via CSV file path or database connection (configure in .env or Section 1 below).

# - Version 1.02  
# - updated 30.04.26

# Imports

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(r'Q:\scripts\projects\ts-model-framework')
import config

print(f"Project path: {config.PROJECT_PATH}")
print(f"Data path: {config.DATA_PATH}")
print(f"Data path: {config.EXPORTS_PATH}")
print("Libraries loaded.")


# ---
# S1 - Data Source Configuration
# Set your data source below. Use either CSV or database connection.
# Comment/uncomment the relevant block.

# %%
# --- Section 1 - Data Source Configuration ---

# ── DATABASE SOURCE (comment out CSV block above if using this) ──
# DB_URL = os.getenv('MLFLOW_DB_URL')              # set in .env
# DB_QUERY = "SELECT * FROM your_table"
# engine = create_engine(DB_URL)
# df = pd.read_sql(DB_QUERY, engine)

# ── LOAD CSV ─────────────────────────────────────────────────
file_path = os.path.join(config.DATA_PATH, config.FILE_NAME)
df = pd.read_csv(file_path, parse_dates=[config.DATE_COLUMN], index_col=config.DATE_COLUMN)

print(f"Loaded: {file_path}")
print(f"Shape: {df.shape}")


# ---
# S2 - Structure & Data Types

# %%
# --- Section 2 - Structure & Data Types
print("Shape:", df.shape)
print()
print("Date range:", df.index.min(), "to", df.index.max())
print()
print("Data types:")
print(df.dtypes)


# ---
# S3 - Missing Values

# %%
# --- Section 3a - Time Series Detection & Date Continuity Check ---

# Detect if index is datetime
is_timeseries = pd.api.types.is_datetime64_any_dtype(df.index)
print(f"Time series data detected: {is_timeseries}")

if is_timeseries:
    full_range    = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    missing_dates = full_range.difference(df.index)

    print(f"Expected dates : {len(full_range)}")
    print(f"Dates present  : {len(df)}")
    print(f"Missing dates  : {len(missing_dates)}")

    if len(missing_dates) > 0:
        print("\nMissing date list:")
        for d in missing_dates:
            print(f"  {d.date()}")
    else:
        print("No missing dates -- date continuity confirmed.")
else:
    print("Non-time series dataset -- skipping date continuity check.")

# %%
# --- Section 3b - Non-Timeseries Missing Value Check ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query("missing_count > 0")

if missing_summary.empty:
    print("No missing values found.")
else:
    print(missing_summary)


# ---
# S4 - Basic Statistics

# %%
# --- Section 4 - Basic Stats ---
df.describe().round(2)


# ---
# S5 - Target Variable Plot

# %%
# --- Section 5 - Target Variable Plot ---
fig, ax = plt.subplots(figsize=(15, 5))
df[config.TARGET_COLUMN].plot(ax=ax)
ax.set_title(f'{config.TARGET_COLUMN} over time')
ax.set_ylabel(config.TARGET_COLUMN)
plt.tight_layout()
plt.show()


# ---
# S6 - Seasonal Decomposition
# Splits the target into trend, seasonality, and residual components.
# Adjust `period` to match your data frequency (7=weekly, 12=monthly, 365=yearly).

# %%
# --- Section 6 - Seasonal Decomp ---
PERIOD = 7   # adjust as needed

decomp = seasonal_decompose(df[config.TARGET_COLUMN].dropna(), model='additive', period=PERIOD)

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title='Observed')
decomp.trend.plot(ax=axes[1], title='Trend')
decomp.seasonal.plot(ax=axes[2], title='Seasonality')
decomp.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()


# ---
# S7 - Notes & Observations
# Use this cell to document findings for this dataset.
# 
# - Date range: 2013-01-02 00:00:00 to 2014-03-31 00:00:00  
# - Key patterns observed:
# - Missing Dates: 2013-12-25, 2014-01-01 -- both holidays, will fill sales data with zeros
# - Missing value decisions:  
# - Recommended period for decomposition:
# - Features worth engineering:


