import gradio as gr
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
MODEL_LOAD_PATH = "lstm_msft_model.keras" # Ensure this file is uploaded to your HF Space
CSV_PATH = "MSFT.csv"                     # Ensure this file is uploaded to your HF Space
RANDOM_STATE = 22
START_DATE_DATA = "2023-01-01"
END_DATE_DATA = "2025-12-31"

FEATURES = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_5", "MA_10", "RSI"]
TARGET_FEATURE = "Close"
MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 10
RSI_WINDOW = 14
LOOK_BACK = 20
TRAIN_SPLIT = 0.8
SCALER_RANGE = (0, 1)

# =========================
# 1. INITIAL SETUP & PROCESSING
# =========================
# We run this once at startup so the app is fast

# Seeding
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Data Loading
print("Loading data...")
try:
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= START_DATE_DATA) & (df["Date"] <= END_DATE_DATA)]
    df.set_index("Date", inplace=True)
except FileNotFoundError:
    raise RuntimeError(f"Could not find {CSV_PATH}. Make sure to upload it to Hugging Face Files.")

# Feature Engineering
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(MA_SHORT_WINDOW).mean()
df["MA_10"] = df["Close"].rolling(MA_LONG_WINDOW).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(RSI_WINDOW).mean() / loss.rolling(RSI_WINDOW).mean()
df["RSI"] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# Scaling
data = df[FEATURES].values
scaler = MinMaxScaler(feature_range=SCALER_RANGE)
scaled_data = scaler.fit_transform(data)
TARGET_COL = FEATURES.index(TARGET_FEATURE)

# Sequence Creation
def create_sequences(dataset, look_back, target_col):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, :])
        y.append(dataset[i + look_back, target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOK_BACK, TARGET_COL)

# Splitting
train_size = int(len(X) * TRAIN_SPLIT)
X_test = X[train_size:]
y_test = y[train_size:]

# Dates Alignment for Test Set
# The test set starts after train_size + look_back
dates = df.index
test_dates = dates[train_size + LOOK_BACK:]

# Model Loading & Prediction
print("Loading model and generating predictions...")
try:
    model = load_model(MODEL_LOAD_PATH)
    test_pred_scaled = model.predict(X_test)
except IOError:
    # Fallback for testing without model file (creates dummy prediction)
    print("Warning: Model file not found. Generating dummy predictions for UI testing.")
    test_pred_scaled = np.zeros((len(X_test), 1))

# Inversion
def invert_close(pred):
    dummy = np.zeros((len(pred), scaled_data.shape[1]))
    dummy[:, TARGET_COL] = pred.flatten()
    return scaler.inverse_transform(dummy)[:, TARGET_COL]

test_pred_actual = invert_close(test_pred_scaled)
y_test_actual = invert_close(y_test.reshape(-1, 1))

# Create a DataFrame for easy filtering
results_df = pd.DataFrame({
    "Date": test_dates,
    "Actual": y_test_actual,
    "Prediction": test_pred_actual
})
results_df.set_index("Date", inplace=True)

# Min/Max dates for UI Helper
MIN_DATE = results_df.index.min().date()
MAX_DATE = results_df.index.max().date()

# =========================
# 2. GRADIO INTERFACE FUNCTION
# =========================

def plot_stock(start_date, end_date):
    # Validate dates
    try:
        s_date = pd.to_datetime(start_date)
        e_date = pd.to_datetime(end_date)
    except:
        return None, "Error: Invalid date format. Use YYYY-MM-DD."
    
    if s_date > e_date:
        return None, "Error: Start date cannot be after end date."

    # Filter data
    mask = (results_df.index >= s_date) & (results_df.index <= e_date)
    filtered_df = results_df.loc[mask]

    if filtered_df.empty:
        return None, f"No data found for this range. Test data is available from {MIN_DATE} to {MAX_DATE}."

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.plot(filtered_df.index, filtered_df["Actual"], label="Actual Price", color="blue", linewidth=2)
    plt.plot(filtered_df.index, filtered_df["Prediction"], label="Predicted Price", color="orange", linestyle="--", linewidth=2)
    
    plt.title(f"MSFT Stock: Actual vs Prediction ({s_date.date()} to {e_date.date()})")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    return fig, f"Showing data for {len(filtered_df)} days."

# =========================
# 3. GRADIO APP LAUNCH
# =========================

with gr.Blocks() as demo:
    gr.Markdown(f"# 📈 MSFT Stock Prediction (LSTM)\nCompare real vs predicted stock prices on unseen test data.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"**Available Test Data Range:**\n{MIN_DATE} to {MAX_DATE}")
            
            inp_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=str(MIN_DATE))
            inp_end = gr.Textbox(label="End Date (YYYY-MM-DD)", value=str(MAX_DATE))
            
            btn = gr.Button("Generate Plot", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Stock Chart")

    btn.click(fn=plot_stock, inputs=[inp_start, inp_end], outputs=[plot_output, status_output])

if __name__ == "__main__":
    demo.launch()