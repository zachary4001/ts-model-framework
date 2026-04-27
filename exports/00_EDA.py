# 00 - Exploratory Data Analysis  
# Reusable EDA notebook for any time series dataset.  
# Load via CSV file path or database connection (configure in .env or Section 1 below).


# ## Imports

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv
from sqlalchemy import create_engine
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
# ## S1 - Data Source Configuration
# Set your data source below. Use either CSV or database connection.
# Comment/uncomment the relevant block.

# %%
# ── CSV SOURCE ──────────────────────────────────────────────
DATA_PATH = os.getenv('DATA_PATH')        # set in .env, or override below
# DATA_PATH = r"Q:\scripts\projects\ts-model-framework\data"  # manual override

FILE_NAME = "timeseries_with_features.csv"   # change as needed
DATE_COLUMN = "date"                          # change to match your dataset
TARGET_COLUMN = "unit_sales"                  # change to match your dataset

# ── DATABASE SOURCE (comment out CSV block above if using this) ──
# DB_URL = os.getenv('MLFLOW_DB_URL')              # set in .env
# DB_QUERY = "SELECT * FROM your_table"
# engine = create_engine(DB_URL)
# df = pd.read_sql(DB_QUERY, engine)

# ── LOAD CSV ─────────────────────────────────────────────────
file_path = os.path.join(DATA_PATH, FILE_NAME)
df = pd.read_csv(file_path, parse_dates=[DATE_COLUMN], index_col=DATE_COLUMN)

print(f"Loaded: {file_path}")
print(f"Shape: {df.shape}")


# ---
# ## S2 - Structure & Data Types

# %%
print("Shape:", df.shape)
print()
print("Date range:", df.index.min(), "to", df.index.max())
print()
print("Data types:")
print(df.dtypes)


# ---
# ## S3 - Missing Values

# %%
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
# ## S4 - Basic Statistics

# %%
df.describe().round(2)


# ---
# ## S5 - Target Variable Plot

# %%
fig, ax = plt.subplots(figsize=(15, 5))
df[TARGET_COLUMN].plot(ax=ax)
ax.set_title(f'{TARGET_COLUMN} over time')
ax.set_ylabel(TARGET_COLUMN)
plt.tight_layout()
plt.show()


# ---
# ## S6 - Seasonal Decomposition
# Splits the target into trend, seasonality, and residual components.
# Adjust `period` to match your data frequency (7=weekly, 12=monthly, 365=yearly).

# %%
PERIOD = 7   # adjust as needed

decomp = seasonal_decompose(df[TARGET_COLUMN].dropna(), model='additive', period=PERIOD)

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title='Observed')
decomp.trend.plot(ax=axes[1], title='Trend')
decomp.seasonal.plot(ax=axes[2], title='Seasonality')
decomp.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()


# ---
# ## S7 - Notes & Observations
# Use this cell to document findings for this dataset.
# 
# - Date range:
# - Key patterns observed:
# - Missing value decisions:
# - Recommended period for decomposition:
# - Features worth engineering:


