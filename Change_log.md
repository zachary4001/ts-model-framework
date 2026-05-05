# Change Log  
- updated 05.05.26  

## General configuration changes  
01.05.2026 - Added config.py file and retrofit all existing files  
30.04.2026 - Added version and last updated to all process files
05.05.2026 - Added 05_residuals.ipynb to process file list

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
- 05_residuals.ipynb
    - outputs to 05_residuals.py
- app.py  
- config.py  


### 00_EDA.ipynb  
05.05.2026  
- Added Average Unit Sales by Day of Week bar chart -- confirms weekly seasonality cycle  
30.04.2026  
- Added section (a) for Time Series to the existing `missing values` section with (b) non-timeseries missing values Detection & Date Continuity Check  

### 01_preprocessing.ipynb  
01.05.2026  
- Prophet preprocessing column rename ('index' -> 'ds') in S3 Prophet Variant  
30.04.2026  `prophet variant`
- Added missing dates detection and reindex to full date range   
- Added fillna(0) for missing dates, commenting line for ffill `load core dataset`  

### 02_experiments.ipynb  
05.05.2026  
- Added `Oil Correlation Analysis` (dual-axis overlay, rolling 30d correlation, scatter + Pearson r)  
- Added `Oil Rescaled Analysis` (z-score overlay, above/below $100 distribution, daily pct change scatter)  
- Added `XGBoost Feature Ablation` loop (4 variants: calendar-only, lag-rolling-only, exogenous-only, no-exogenous)  
- Added `XGBoost Feature Importance` chart (gain-based, saved to models folder)  
- Added `Holt-Winters` Exponential Smoothing Baseline (seasonal_periods=7, logs as `holtwinters-baseline`)  
- Added `Prophet Tuning`: 2 predefined variations + 24-combo grid search; champion: prophet-v1-multiplicative (RMSE 150.33)  
- Added `SARIMAX No-Oil grid search` (holidays-only exog); best order (1,1,2), logs as `sarimax-no-oil`  
- Added `XGBoost Hyperopt tuned`, no-exogenous features (logs as `xgb-no-exo-tuned`)  
- Added `XGBoost focused` feature experiments loop (4 variants: minimal-day-lag1, minimal-D-W-Q-lag1, minimal-calendar,  top11-features)  
- Added `SARIMAX No-Oil` grid search cell (replaces hardcoded order version)  
- Added `Save Classical & ML Model` Predictions to CSV (xgboost, sarimax, prophet, holtwinters _predictions.csv)  
    - for use in `05_residuals` and other potential future uses.  
- Fixed ablation MLflow logging -- added feature_list, n_estimators, tuned=False params  
- Fixed `prophet` index alignment -- switched from y_test.index to prophet_test source  
- Fixed MAPE division-by-zero on zero-sales dates in `shared evaluate function` -- added numpy mask (`actuals != 0`) before calculation   
- Fixed `Results Summary` -- replaced hardcoded experiment name string with `config.EXPERIMENT` variable  
- Added `XGBoost Hyperopt tuning` (20 trials, logs as `xgboost-tuned`)  
- Added `SARIMAX grid search` across p∈[0,1,2], d∈[0,1], q∈[0,1,2]; best order (1,1,2), logs as `sarimax-tuned`  
01.05.2026  
- Fixed save logic to select best model from local results only (not MLflow) `Save Classical & ML Model`   
- Added per-model metric capture (metrics_xgb, metrics_sarimax, metrics_prophet)  
30.04.2026  
- Added Bias metric to `shared evaluate function`  
- Added `Save Best Model` (joblib for classical models, best_model_name.txt for Streamlit)  
- Added MODELS_PATH to `imports` & global variables section  

### 03_LSTM.ipynb   
05.05.2026  
- Added `Save Predictions` for Residuals Analysis (lstm_predictions.csv to models folder)  
- Added `Sequence Length Experiment Loop` testing seq_len ∈ [14, 60] (addition to existing 30) with full Hyperopt tuning per variant; logs as `lstm-seq{n}-tuned`  
01.05.2026  
- Added `Save Tuned LSTM Model` section - Save Tuned LSTM Model (torch.save state_dict to models/best_lstm_model.pt)    
30.04.2026  
- Added `Hyperopt` Tuning and retrain cells  
- Added random seeds for torch and np `import`  
- Added early stopping to training loop `Train the model`  

### 04_RNN.ipynb  
05.05.2026  
- Added `Save Predictions` for Residuals Analysis (rnn_predictions.csv to models folder)  
- Added `Sequence length` experiment loop seq_len ∈ [14, 60]  
- Added `Sequence Length Experiment Loop` testing seq_len ∈ [14, 60] (addition to existing 30) with full Hyperopt tuning per variant; logs as `rnn-seq{n}-tuned`  
01.05.2026  
- Added `Save Tuned LSTM Model` section - Save Tuned RNN Model (torch.save state_dict to models/best_rnn_model.pt)  

### 05_residuals.ipynb (new)
05.05.2026
- Created new notebook for cross-model residuals analysis
- Loads all 6 model prediction CSVs from models folder
- Residuals over time (all models, consistent y-axis scale)
- Error distribution histograms (all models, consistent y-axis, fixed bin width)
- Actual vs Predicted scatter (all models, consistent axes)
- ACF of residuals (all models, 30 lags)
- Summary table (RMSE, MAE, Bias, Std Err)
- Actual vs Predicted overlay line chart (RNN champion, with error band)
- Average Unit Sales by Day of Week bar chart (moved from EDA for residuals context)

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

### experiments list
05.05.2026 - Changed EXPERIMENT name from `ts-model-framework` to `favorita` in config.py  
