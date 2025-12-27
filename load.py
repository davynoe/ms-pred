# =========================
# IMPORT
# =========================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# =========================
# CONFIGURATION
# =========================

# ---- Model Loading ----
MODEL_LOAD_PATH = "lstm_msft_model.keras"

# ---- Reproducibility ----
RANDOM_STATE = 22

# ---- Data ----
CSV_PATH = "MSFT.csv"
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# ---- Features ----
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA_5", "MA_10", "RSI"
]
TARGET_FEATURE = "Close"

# ---- Feature Engineering ----
MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 10
RSI_WINDOW = 14

# ---- Scaling ----
SCALER_RANGE = (0, 1)

# ---- Sequence ----
LOOK_BACK = 20
TRAIN_SPLIT = 0.8

# ---- Plotting ----
FIGSIZE_MAIN = (14, 7)
FIGSIZE_LOSS = (12, 5)

# =========================
# SEEDING (FOR CONSISTENCY)
# =========================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================
# DATA LOADING
# =========================

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
df.set_index("Date", inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================

df["Return"] = df["Close"].pct_change()

df["MA_5"] = df["Close"].rolling(MA_SHORT_WINDOW).mean()
df["MA_10"] = df["Close"].rolling(MA_LONG_WINDOW).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(RSI_WINDOW).mean() / loss.rolling(RSI_WINDOW).mean()
df["RSI"] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# =========================
# SCALING
# =========================

data = df[FEATURES].values

scaler = MinMaxScaler(feature_range=SCALER_RANGE)
scaled_data = scaler.fit_transform(data)

TARGET_COL = FEATURES.index(TARGET_FEATURE)

# =========================
# SEQUENCE CREATION
# =========================

def create_sequences(dataset, look_back, target_col):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, :])
        y.append(dataset[i + look_back, target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOK_BACK, TARGET_COL)

train_size = int(len(X) * TRAIN_SPLIT)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# LOAD MODEL
# =========================

model = load_model(MODEL_LOAD_PATH)
model.summary()

# =========================
# PREDICTION & INVERSION
# =========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

def invert_close(pred):
    dummy = np.zeros((len(pred), scaled_data.shape[1]))
    dummy[:, TARGET_COL] = pred.flatten()
    return scaler.inverse_transform(dummy)[:, TARGET_COL]

train_pred = invert_close(train_pred)
test_pred = invert_close(test_pred)

# =========================
# DATES ALIGNMENT
# =========================

dates = df.index
train_dates = dates[LOOK_BACK : LOOK_BACK + len(train_pred)]
test_dates = dates[
    LOOK_BACK + len(train_pred) :
    LOOK_BACK + len(train_pred) + len(test_pred)
]

# =========================
# OUTPUT
# =========================

print("\n===== MODEL INFERENCE MODE =====")
print(f"Loaded model from : {MODEL_LOAD_PATH}")
print(f"Random seed used  : {RANDOM_STATE}")
print("================================")

# =========================
# PLOTTING
# =========================

plt.figure(figsize=FIGSIZE_MAIN)
plt.plot(dates, df["Close"], label="Actual Price", color="lightgray", linewidth=2)
plt.plot(train_dates, train_pred, label="Train Prediction", linestyle="--")
plt.plot(test_dates, test_pred, label="Test Prediction")
plt.title("MSFT Stock Price Prediction (LSTM) – Loaded Model")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
