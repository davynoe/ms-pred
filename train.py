# =========================
# IMPORT
# =========================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# =========================
# CONFIGURATION
# =========================

# ---- Model Saving ----
MODEL_SAVE_PATH = "lstm_msft_model_temp.keras"

# ---- Plot Saving ----
PRICE_PLOT_PATH = "msft_lstm_price_prediction.png"
LOSS_PLOT_PATH = "msft_lstm_training_loss.png"
PLOT_DPI = 300
PLOT_BBOX = "tight"

# ---- Reproducibility ----
RANDOM_STATE = 46

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

# ---- Model Architecture ----
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2

# ---- Training ----
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ---- Early Stopping ----
EARLY_STOP_MONITOR = "val_loss"
EARLY_STOP_PATIENCE = 10
RESTORE_BEST_WEIGHTS = True

# ---- Plotting ----
FIGSIZE_MAIN = (14, 7)
FIGSIZE_LOSS = (12, 5)

# =========================
# SEEDING
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
# MODEL
# =========================

model = Sequential([
    LSTM(
        LSTM_UNITS_1,
        return_sequences=True,
        input_shape=(LOOK_BACK, X.shape[2])
    ),
    Dropout(DROPOUT_RATE),

    LSTM(LSTM_UNITS_2),
    Dropout(DROPOUT_RATE),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse"
)

model.summary()

# =========================
# TRAINING
# =========================

early_stop = EarlyStopping(
    monitor=EARLY_STOP_MONITOR,
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=RESTORE_BEST_WEIGHTS
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

model.save(MODEL_SAVE_PATH)

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

print("\n===== REPRODUCIBILITY INFO =====")
print(f"Python random seed     : {RANDOM_STATE}")
print(f"NumPy random seed      : {RANDOM_STATE}")
print(f"TensorFlow random seed : {RANDOM_STATE}")
print(f"Model saved to         : {MODEL_SAVE_PATH}")
print("================================")

# =========================
# PLOTTING (EXPORT PNG)
# =========================

plt.figure(figsize=FIGSIZE_MAIN)
plt.plot(dates, df["Close"], label="Actual Price", color="lightgray", linewidth=2)
plt.plot(train_dates, train_pred, label="Train Prediction", linestyle="--")
plt.plot(test_dates, test_pred, label="Test Prediction")
plt.title("MSFT Stock Price Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()

plt.savefig(PRICE_PLOT_PATH, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX)
plt.show()

plt.figure(figsize=FIGSIZE_LOSS)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

plt.savefig(LOSS_PLOT_PATH, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX)
plt.show()
