# src/predict.py

import pandas as pd
import joblib
import yfinance as yf
import ta

MODEL_PATH = "models/xgboost_stock_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
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


def get_recommendation(prediction):
    recommendations = {
        0: "SELL",
        1: "HOLD",
        2: "BUY"
    }
    return recommendations[prediction]


def predict_stock(stock_symbol):
    print(f"\nFetching latest data for {stock_symbol}...\n")

    df = yf.download(
        stock_symbol,
        period="6mo",
        interval="1d",
        auto_adjust=True
    )

    if df.empty:
        print("No data found.")
        return

    # Technical Indicators
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["BB_High"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"])
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    df.dropna(inplace=True)

    latest_data = df[FEATURE_COLUMNS].iloc[-1:]

    prediction = model.predict(latest_data)[0]
    probabilities = model.predict_proba(latest_data)[0]

    print("=" * 50)
    print(f"Stock: {stock_symbol}")
    print(f"Current Price: ₹{df['Close'].iloc[-1]:.2f}")
    print(f"Recommendation: {get_recommendation(prediction)}")
    print("\nConfidence Scores:")
    print(f"SELL : {probabilities[0]:.2%}")
    print(f"HOLD : {probabilities[1]:.2%}")
    print(f"BUY  : {probabilities[2]:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    stock = input("Enter Stock Symbol (Example: RELIANCE.NS): ").upper()
    predict_stock(stock)