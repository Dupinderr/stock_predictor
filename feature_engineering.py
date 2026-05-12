import os
import pandas as pd
import numpy as np

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/features"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = os.listdir(INPUT_FOLDER)

for file in files:
    if not file.endswith(".csv"):
        continue

    path = os.path.join(INPUT_FOLDER, file)
    df = pd.read_csv(path)

    print(f"Processing {file}...")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # ================= FEATURES =================

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_High"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Low"] = df["BB_Mid"] - 2 * df["BB_Std"]

    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    # ============================================================
    # 🔥 FIXED LABELING (THIS FIXES HOLD-ONLY PROBLEM)
    # ============================================================

    df["Future_Return"] = df["Close"].shift(-5) / df["Close"] - 1

    buy_threshold = df["Future_Return"].quantile(0.80)
    sell_threshold = df["Future_Return"].quantile(0.20)

    df["Signal"] = 1  # HOLD

    df.loc[df["Future_Return"] >= buy_threshold, "Signal"] = 2  # BUY
    df.loc[df["Future_Return"] <= sell_threshold, "Signal"] = 0  # SELL

    df.dropna(inplace=True)

    # Save
    output_path = os.path.join(OUTPUT_FOLDER, file)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

print("\nFeature Engineering Completed")