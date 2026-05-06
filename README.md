# Retail Sales Forecasting -- Corporación Favorita
**Masterschool MSIT Term 8 | Time Series Modeling Capstone**

A reusable, end-to-end time series forecasting framework applied to retail demand
forecasting using data from the Corporación Favorita grocery chain (Ecuador).
Built as both a working solution and a reusable pipeline template for future
time series projects.

---

## Business Problem

Accurate sales forecasting enables better inventory decisions -- reducing overstock
waste and stockout risk. This project forecasts daily unit sales for a single
store and product as a proof of concept, with the pipeline designed to scale
across stores, products, and business domains (staffing, capital planning, etc.).

**Dataset:** Corporación Favorita | Store 44 (Quito, Pichincha) | Item 1047679
**Training period:** 2013-01-02 to 2013-12-31 (364 days)
**Test period:** 2014-01-01 to 2014-03-31 (90 days)

---

## Project Structure

```
ts-model-framework/
├── data/
│   ├── timeseries_with_features.csv   # merged, feature-engineered dataset
│   ├── timeseries.csv                 # raw sales data
│   ├── oil.csv                        # daily WTI oil prices
│   ├── holidays.csv                   # Ecuador national/regional/local holidays
│   └── stores.csv                     # store metadata
├── models/                            # saved model artifacts + prediction CSVs
├── exports/                           # stable .py exports for notebook imports
│   ├── 00_EDA.py                         # exploratory data analysis
│   ├── 01_preprocessing.py               # feature engineering + train/test split
│   ├── 02_experiments.py                 # classical + XGBoost models + MLflow logging
│   ├── 03_LSTM.py                        # LSTM model + hyperopt tuning
│   ├── 04_RNN.py                         # RNN model + hyperopt tuning
│   └── 05_residuals.py                   # cross-model residuals analysis
├── notebooks/                         # jupyter notebooks for limited change testing
│   ├── 00_EDA.ipynb                      # working source of 00_EDA.py
│   ├── 01_preprocessing.ipynb            # working source of 01_preprocessing.py
│   ├── 02_experiments.ipynb              # working source of 02_experiments.py
│   ├── 03_LSTM.ipynb                     # working source of 03_LSTM.py
│   ├── 04_RNN.ipynb                      # working source of 04_RNN.py
│   └── 05_residuals.ipynb                # working source of 05_residuals.py
├── app.py                             # Streamlit forecasting dashboard
├── config.py                          # single source of truth for all paths/vars
└── requirements.txt                   # required library and versions
```

---

## Models Compared

| Model | Type | Best RMSE |
|---|---|---|
| RNN (tuned) | Deep Learning | 104.46 |
| LSTM (tuned) | Deep Learning | 111.07 |
| XGBoost (tuned) | Machine Learning | 141.87 |
| Prophet (v1-multiplicative) | Statistical | 150.33 |
| Holt-Winters | Statistical | 150.41 |
| SARIMAX (tuned) | Statistical | 185.82 |

**Champion model:** RNN tuned -- consistent winner across all runs, outperforming
the next best model by 30+ RMSE units. Architecture mattered more than tuning
on this small dataset.

---

## Key Findings

- **Weekly cycle dominates:** Sunday averages 694 units vs Thursday 360 -- the
  strongest and most actionable signal in the data
- **Oil prices excluded:** Pearson r = 0.144, rolling correlation unstable,
  above/below $100 threshold negligible -- data overruled initial hypothesis
- **Feature importance:** `dayofweek` contributes 10x more gain than any other
  feature in XGBoost -- confirmed by ablation experiments
- **Sequence length matters:** LSTM/RNN performance drops sharply below 30-day
  sequence windows -- weekly retail patterns require sufficient history context
- **Tuning ceiling:** with 364 training rows, hyperparameter tuning produced
  marginal gains -- more data would unlock greater improvement

---

## Pipeline Architecture

The framework is designed to be reusable across datasets with minimal changes:

1. **`config.py`** -- change paths, dataset name, target column here only
2. **`00_EDA.py`** -- explore raw data, document findings
3. **`01_preprocessing.py`** -- merge sources, engineer features, split data
4. **`02_experiments.py`** -- run all classical/ML models, log to MLflow
5. **`03_LSTM.py`** + **`04_RNN.py`** -- deep learning models with hyperopt
6. **`05_residuals.py`** -- cross-model diagnostics and visualization
7. **`app.py`** -- Streamlit dashboard consuming saved models

---

## Setup & Running Locally

### Requirements
Python 3.11 | See `requirements.txt` for full dependency list

### Install dependencies
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Start MLflow server
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  -h 0.0.0.0 -p 5000
```
*Local Postgres backend also supported -- see config.py comments*

### Run notebooks in order
```
00_EDA → 01_preprocessing → 02_experiments → 03_LSTM → 04_RNN → 05_residuals
```

### Launch Streamlit app
```powershell
cd Q:\scripts\projects\ts-model-framework
streamlit run app.py
```
App available at `http://localhost:8501`

---

## Experiment Tracking

All model runs logged to MLflow. Key experiment: **`favorita`**

Tracked per run:
- Hyperparameters (model type, tuning method, feature set, sequence length)
- Metrics: RMSE, MAE, MAPE, R², Bias
- Model artifacts (saved for Streamlit loading)

---

## Next Steps

- Expand to all 54 stores and full product catalog
- Add Sales Drivers tab: promotions, weather, seasonal events, scenario modeling
- Ensemble: combine RNN + XGBoost predictions
- Add forecast confidence intervals to Streamlit chart
- Apply pipeline to staffing and capital planning forecasting

---

## Author

Jason Zachary Guest | Masterschool MSIT Term 8 | May 2026
