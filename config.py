# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Paths
PROJECT_PATH  = os.getenv('PROJECT_PATH',  r'Q:\\scripts\\projects\\ts-model-framework')
DATA_PATH     = os.getenv('DATA_PATH',     os.path.join(PROJECT_PATH, 'data'))
MODELS_PATH   = os.getenv('MODELS_PATH',   os.path.join(PROJECT_PATH, 'models'))
EXPORTS_PATH  = os.getenv('EXPORTS_PATH',  os.path.join(PROJECT_PATH, 'exports'))

# Dataset
FILE_NAME     = "timeseries_with_features.csv"
DATE_COLUMN   = "date"
TARGET_COLUMN = "unit_sales"

# MLflow
MLFLOW_URI    = os.getenv('MLFLOW_URI', 'http://localhost:5000')
EXPERIMENT    = "ts-model-framework"
