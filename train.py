import os
import glob
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# LOAD DATA
# ============================================================

files = glob.glob("data/features/*.csv")

if not files:
    raise Exception("Run feature_engineering.py first")

data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI",
    "MACD", "MACD_Signal",
    "BB_High", "BB_Low",
    "Daily_Return", "Volatility"
]

X = data[FEATURE_COLUMNS]
y = data["Signal"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nClass Distribution:")
print(y_train.value_counts())

# ============================================================
# FINAL TUNED XGBOOST (STABLE VERSION)
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# ============================================================
# TRAIN
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# EVALUATE
# ============================================================

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")

print("\n================ RESULTS ================\n")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_stock_model.pkl")

print("\nModel saved successfully")