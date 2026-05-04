# config.py
# this config file is for the project Corporacion Favorita retail data
# updated 04.05.2026

import os
from dotenv import load_dotenv
load_dotenv()

# Detect if running on Streamlit Cloud or locally
IS_LOCAL = os.path.exists(r'Q:\scripts\projects\ts-model-framework')
if IS_LOCAL:
    PROJECT_PATH = r'Q:\scripts\projects\ts-model-framework'
else:
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH     = os.getenv('DATA_PATH',     os.path.join(PROJECT_PATH, 'data'))
MODELS_PATH   = os.getenv('MODELS_PATH',   os.path.join(PROJECT_PATH, 'models'))
EXPORTS_PATH  = os.getenv('EXPORTS_PATH',  os.path.join(PROJECT_PATH, 'exports'))

# Dataset
FILE_NAME     = "timeseries_with_features.csv"
DATE_COLUMN   = "date"
TARGET_COLUMN = "unit_sales"

# MLflow
MLFLOW_URI    = os.getenv('MLFLOW_URI', 'http://localhost:5000')
EXPERIMENT    = "favorita"