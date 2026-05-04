# Change Log  
- updated 30.04.26  

## General configuration changes  
01.05.2026 - Added config.py file and retrofit all existing files  
30.04.2026 - Added version and last updated to all process files


## Specific File or Process Step changes  
Current Process Files:  
- 00_EDA.ipynb  
- 01_preprocessing.ipynb  
    - outputs to 01_preprocessing.py  
- 02_experiments.ipynb  
    - outputs to 02_experiments.py  
- 03_LSTM.ipynb  
    - outputs to 03_LSTM.py  
- 04_RNN.ipynb  
    - outputs to 04_RNN.py  
- app.py  
- config.py  


### 00_EDA.ipynb  
30.04.2026  
- Added section (a) for Time Series to the existing `missing values` section with (b) non-timeseries missing values Detection & Date Continuity Check  

### 01_preprocessing.ipynb  
01.05.2026  
- Prophet preprocessing column rename ('index' -> 'ds') in S3 Prophet Variant  
30.04.2026  `prophet variant`
- Added missing dates detection and reindex to full date range   
- Added fillna(0) for missing dates, commenting line for ffill `load core dataset`  

### 02_experiments.ipynb  
01.05.2026  
- Fixed S8 save logic to select best model from local results only (not MLflow)  
- Added per-model metric capture (metrics_xgb, metrics_sarimax, metrics_prophet)  
30.04.2026  
- Added Bias metric to `shared evaluate function`  
- Added `Save Best Model` (joblib for classical models, best_model_name.txt for Streamlit)  
- Added MODELS_PATH to `imports` & global variables section  

### 03_LSTM.ipynb   
01.05.2026  
- Added `Save Tuned LSTM Model` section - Save Tuned LSTM Model (torch.save state_dict to models/best_lstm_model.pt)    
30.04.2026  
- Added `Hyperopt` Tuning and retrain cells  
- Added random seeds for torch and np `import`  
- Added early stopping to training loop `Train the model`  

### 04_RNN.ipynb  
01.05.2026  
- Added `Save Tuned LSTM Model` section - Save Tuned RNN Model (torch.save state_dict to models/best_rnn_model.pt)  

### app.py  
01.05.2026  
- Migrated all path/config variables to config.py  
    - this better standardized global variablescorrected  
    - Specifically fixing a problem with streamlit using variables from .env  
- Fixed load_mlflow_results() -- corrected search_runs() to use experiment_names= keyword  
- Removed @st.cache_resource from load_model() to prevent stale None caching  
- Added @st.cache_data(ttl=60) to load_mlflow_results() - setting a 60 second cache timer   
- Fixed "model not found error" caused by single backslashes in config.py path joins  

### config.py  
01.05.2026  
- Created as project-wide variable store replacing .env direct loading in all files  
- Fixed path join fallbacks -- removed leading backslash from subfolder strings  
