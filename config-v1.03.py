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

# DELETE everything BELOW before saving to 'config.py' and upload to github
# To activate MLflow server manually, run the command below from powershell terminal
# > mlflow server --backend-store-uri 'postgresql://postgres_admin:9r&#2h$5y^6b78n8pU*jF8@192.168.2.30:5432/mlflow_tracking' --default-artifact-root './mlartifacts' -h 0.0.0.0 -p 5000

# To run streamlet against Best model, run the commands below from powershell terminal
# cd Q:\scripts\projects\ts-model-framework
# streamlit run app.py