# 03 - LSTM Forecasting Model


# - Version 1.03  
# - updated 04.05.26


# Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN)
# designed to learn patterns across time sequences.
# 
# Unlike XGBoost which sees each row independently, LSTM sees a *window* of
# past timesteps and learns how the sequence evolves over time.
# 
# Key concepts:
# - SEQUENCE_LENGTH: how many past days the model looks at to make one prediction
# - Scaling: LSTM requires values between 0-1 (MinMaxScaler)
# - GPU: training runs on RTX 2060 automatically if CUDA is available


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import mlflow
import sys
sys.path.append(r'Q:\scripts\projects\ts-model-framework')
import config

import warnings
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mlflow.set_tracking_uri(config.MLFLOW_URI)
mlflow.set_experiment(config.EXPERIMENT)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries loaded.")


# ---
## S1 - Load & Scale Data
# LSTM requires values scaled to 0-1 range.
# Scaler is fit on training data only -- applied to test to prevent data leakage.


# --- Section 1 - Load & Scale the Data ---
# FILE_NAME     = "timeseries_with_features.csv"
# DATE_COLUMN   = "date"
# TARGET_COLUMN = "unit_sales"
TEST_START    = "2014-01-01"

file_path = os.path.join(config.DATA_PATH, config.FILE_NAME)
df = pd.read_csv(file_path, parse_dates=[config.DATE_COLUMN], index_col=config.DATE_COLUMN)
df = df.sort_index()

# Split
df_train = df[df.index < TEST_START][[config.TARGET_COLUMN]].fillna(method='ffill')
df_test  = df[df.index >= TEST_START][[config.TARGET_COLUMN]].fillna(method='ffill')

# Scale
scaler      = MinMaxScaler()
train_scaled = scaler.fit_transform(df_train)
test_scaled  = scaler.transform(df_test)

print(f"Train: {df_train.shape} | Test: {df_test.shape}")
print(f"Scaled range -- min: {train_scaled.min():.3f} max: {train_scaled.max():.3f}")


# ---
## S2 - Build Sequences
# LSTM doesn't see individual rows -- it sees sliding windows of past values.
# 
# Example with SEQUENCE_LENGTH=7:
# - Input:  [day1, day2, day3, day4, day5, day6, day7]
# - Target: [day8]
# - Next:   [day2, day3, day4, day5, day6, day7, day8] → [day9]


# --- Section 2 - Build Sequences ---
SEQUENCE_LENGTH = 30   # days of history per prediction

def make_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train_np, y_train_np = make_sequences(train_scaled, SEQUENCE_LENGTH)
X_test_np,  y_test_np  = make_sequences(test_scaled,  SEQUENCE_LENGTH)

# Reshape for PyTorch: (samples, timesteps, features)
X_train_np = X_train_np.reshape(-1, SEQUENCE_LENGTH, 1)
X_test_np  = X_test_np.reshape(-1, SEQUENCE_LENGTH, 1)

print(f"X_train: {X_train_np.shape} | y_train: {y_train_np.shape}")
print(f"X_test:  {X_test_np.shape}  | y_test:  {y_test_np.shape}")


# ---
## S3 - PyTorch Dataset & DataLoader
# PyTorch requires data wrapped in Tensors and fed through a DataLoader.
# BATCH_SIZE controls how many sequences are processed per training step.


# --- Section 3 - PyTorch Dataset & DataLoader ---
BATCH_SIZE = 32

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_np).to(device)
y_train_t = torch.FloatTensor(y_train_np).to(device)
X_test_t  = torch.FloatTensor(X_test_np).to(device)
y_test_t  = torch.FloatTensor(y_test_np).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches per epoch: {len(train_loader)}")
print(f"Device: {device}")


# ---
## S4 - Define LSTM Model
# HIDDEN_SIZE: number of memory units in each LSTM layer
# NUM_LAYERS:  how many LSTM layers stacked on top of each other
# DROPOUT:     randomly disables neurons during training to prevent overfitting


# --- Section 4 - Define LSTM Model ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 50
LEARNING_RATE = 0.001

model     = LSTMForecaster(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")


# ---
## S5 - Train the Model
# Loss should decrease each epoch -- if it stops improving, training is converging.


# --- Section 5 - Train the Model ---
train_losses  = []
best_loss     = float('inf')
patience      = 15
patience_count = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds  = model(X_batch).squeeze()
        loss   = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss      = avg_loss
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

print("Training complete.")


# ---
## S6 - Evaluate & Log to MLflow


# --- Section 6 - Evaluate & Log to MLflow ---
params = {
    "sequence_length": SEQUENCE_LENGTH,
    "hidden_size":     HIDDEN_SIZE,
    "num_layers":      NUM_LAYERS,
    "dropout":         DROPOUT,
    "epochs":          EPOCHS,
    "learning_rate":   LEARNING_RATE,
    "batch_size":      BATCH_SIZE
}

with mlflow.start_run(run_name="lstm-baseline"):
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).squeeze().cpu().numpy()

    # Inverse transform back to original scale
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    r2   = 1 - (np.sum((actuals - preds)**2) / np.sum((actuals - np.mean(actuals))**2))

    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape, "r2": r2})

    print(f"LSTM Baseline Results:")
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")


# ---
## S7 - Hyperopt Tuning 


# --- Section 7a — Hyperopt Tuning ---
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

search_space = {
    'hidden_size':   hp.choice('hidden_size',  [32, 64, 128]),
    'learning_rate': hp.loguniform('lr',       np.log(1e-4), np.log(1e-2)),
    'batch_size':    hp.choice('batch_size',   [8, 16, 32]),
}

def objective(params):
    hidden_size   = params['hidden_size']
    learning_rate = params['learning_rate']
    batch_size    = int(params['batch_size'])

    trial_model  = LSTMForecaster(hidden_size=hidden_size).to(device)
    trial_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=False
    )

    criterion_h  = nn.MSELoss()
    optimizer_h  = torch.optim.Adam(trial_model.parameters(), lr=learning_rate)
    best_loss    = float('inf')
    patience_count = 0

    for epoch in range(100):
        trial_model.train()
        epoch_loss = 0
        for X_batch, y_batch in trial_loader:
            optimizer_h.zero_grad()
            preds = trial_model(X_batch).squeeze()
            loss  = criterion_h(preds, y_batch)
            loss.backward()
            optimizer_h.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(trial_loader)

        if avg_loss < best_loss:
            best_loss      = avg_loss
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= 10:
            break

    return {'loss': best_loss, 'status': STATUS_OK, 'params': params}

trials     = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                   max_evals=20, trials=trials)

print(f"\nBest params found:")
print(f"  hidden_size:   {[32, 64, 128][best_params['hidden_size']]}")
print(f"  learning_rate: {best_params['lr']:.6f}")
print(f"  batch_size:    {[8, 16, 32][best_params['batch_size']]}")


# --- Section 7b — Retrain & Log Tuned LSTM ---
tuned_params = {
    "hidden_size":   [32, 64, 128][best_params['hidden_size']],
    "learning_rate": best_params['lr'],
    "batch_size":    [8, 16, 32][best_params['batch_size']],
    "sequence_length": SEQUENCE_LENGTH,
    "epochs": EPOCHS
}

tuned_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=tuned_params['batch_size'],
    shuffle=False
)

tuned_model   = LSTMForecaster(hidden_size=tuned_params['hidden_size']).to(device)
tuned_criterion = nn.MSELoss()
tuned_optimizer = torch.optim.Adam(tuned_model.parameters(), lr=tuned_params['learning_rate'])

# Train
best_loss, patience_count = float('inf'), 0
for epoch in range(EPOCHS):
    tuned_model.train()
    epoch_loss = 0
    for X_batch, y_batch in tuned_loader:
        tuned_optimizer.zero_grad()
        preds = tuned_model(X_batch).squeeze()
        loss  = tuned_criterion(preds, y_batch)
        loss.backward()
        tuned_optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(tuned_loader)
    if avg_loss < best_loss:
        best_loss, patience_count = avg_loss, 0
    else:
        patience_count += 1
    if patience_count >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Evaluate & log
with mlflow.start_run(run_name="lstm-tuned"):
    tuned_model.eval()
    with torch.no_grad():
        preds_scaled = tuned_model(X_test_t).squeeze().cpu().numpy()

    preds   = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    r2   = 1 - (np.sum((actuals - preds)**2) / np.sum((actuals - np.mean(actuals))**2))
    bias = np.mean(preds - actuals)

    mlflow.log_params(tuned_params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "bias": bias})

    print(f"LSTM Tuned Results:")
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f} | Bias: {bias:.2f}")


# ---
## S8 - Save Tuned LSTM Model


# S8 - Save Tuned LSTM Model
import torch

lstm_save_path = os.path.join(config.MODELS_PATH, 'best_lstm_model.pt')
torch.save(tuned_model.state_dict(), lstm_save_path)

# Save model name
name_path = os.path.join(config.MODELS_PATH, 'best_lstm_model_name.txt')
with open(name_path, 'w') as f:
    f.write(f"lstm-tuned | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

print(f"LSTM model saved: {lstm_save_path}")
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")


# ---
## S9 - Training Loss & Forecast Plot


# --- S9 - Training Loss & Forecast Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Training loss curve
axes[0].plot(train_losses)
axes[0].set_title('Training Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')

# Forecast vs actuals
test_dates = df_test.index[SEQUENCE_LENGTH:]
axes[1].plot(test_dates, actuals, label='Actual', linewidth=2)
axes[1].plot(test_dates, preds,   label='LSTM Forecast', linestyle='--')
axes[1].set_title('LSTM Forecast vs Actuals')
axes[1].set_ylabel(config.TARGET_COLUMN)
axes[1].legend()

plt.tight_layout()
plt.show()


# ---
## S10 - Notes & Observations
# 
# - SEQUENCE_LENGTH chosen:
# - Loss converged around epoch:
# - RMSE vs XGBoost baseline (144.53):
# - RMSE vs SARIMAX baseline (142.02):
# - GPU training time approx:
# - Next experiment to try:


