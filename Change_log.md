# Change Log  
- updated 30.04.26  

## General configuration changes  
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


### 00_EDA.ipynb  
30.04.2026  
- Added section (a) for Time Series to the existing `missing values` section with (b) non-timeseries missing values Detection & Date Continuity Check  

### 01_preprocessing.ipynb  
30.04.2026  
- Added missing dates detection and reindex to full date range   
- Added fillna(0) for missing dates, commenting line for ffill `load core dataset`  

### 02_experiments.ipynb  
30.04.2026  
- Added Bias metric to `shared evaluate function`  
- Added `Save Best Model` (joblib for classical models, best_model_name.txt for Streamlit)  
- Added MODELS_PATH to `imports` & global variables section  

### 03_LSTM.ipynb  
30.04.2026  
- Added `Hyperopt` Tuning and retrain cells  
- Added random seeds for torch and np `import`  
- Added early stopping to training loop `Train the model`  



