# src/evaluate.py

import pandas as pd
import glob
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

FEATURE_PATH = "data/features/"
MODEL_PATH = "models/xgboost_stock_model.pkl"

model = joblib.load(MODEL_PATH)

csv_files = glob.glob(os.path.join(FEATURE_PATH, "*_features.csv"))

all_data = []

for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "BB_High",
    "BB_Low",
    "Daily_Return",
    "Volatility"
]

X = data[features]
y = data["Signal"]

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print(f"\nOverall Accuracy: {accuracy:.2%}\n")

print(classification_report(
    y,
    predictions,
    target_names=["Sell", "Hold", "Buy"]
))

cm = confusion_matrix(y, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Sell", "Hold", "Buy"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()