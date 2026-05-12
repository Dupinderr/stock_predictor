import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import ta
import plotly.graph_objects as go
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Indian Stock Recommendation System",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_PATH = "models/xgboost_stock_model.pkl"

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

RECOMMENDATIONS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}

# ============================================================
# STOCK DROPDOWN
# ============================================================
STOCK_OPTIONS = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "BEL": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS",
    "Grasim": "GRASIM.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "Jio Financial": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Life": "SBILIFE.NS",
    "State Bank of India": "SBIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Wipro": "WIPRO.NS"
}

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run: python src/train.py")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def prepare_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols].copy()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Moving Averages
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # RSI
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    # Returns and Volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# ============================================================
# USER INTERFACE
# ============================================================
st.title("📈 Indian Stock Recommendation System")
st.markdown("### AI-Powered Buy / Hold / Sell Predictions")

selected_stock = st.selectbox(
    "Select an Indian Stock",
    list(STOCK_OPTIONS.keys())
)

ticker = STOCK_OPTIONS[selected_stock]

# ============================================================
# ANALYSIS BUTTON
# ============================================================
if st.button("Analyze Stock"):

    try:
        with st.spinner("Fetching latest stock data..."):
            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

        if df.empty:
            st.error("Unable to fetch stock data.")
            st.stop()

        df = prepare_features(df)

        latest_features = df[FEATURE_COLUMNS].iloc[[-1]]

        prediction = model.predict(latest_features)[0]
        probabilities = model.predict_proba(latest_features)[0]

        recommendation = RECOMMENDATIONS[prediction]
        confidence = np.max(probabilities) * 100
        current_price = float(df["Close"].iloc[-1])

        # ====================================================
        # RESULTS
        # ====================================================
        st.success(f"Recommendation: {recommendation}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Stock", selected_stock)

        with col2:
            st.metric("Current Price", f"₹{current_price:,.2f}")

        with col3:
            st.metric("Confidence", f"{confidence:.2f}%")

        # ====================================================
        # PROBABILITY CHART
        # ====================================================
        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Recommendation": ["SELL", "HOLD", "BUY"],
            "Probability": probabilities * 100
        })

        st.bar_chart(
            prob_df.set_index("Recommendation")
        )

        # ====================================================
        # CANDLESTICK CHART
        # ====================================================
        st.subheader(f"{selected_stock} Price Chart")

        chart_df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.get_level_values(0)

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["Open"],
                    high=chart_df["High"],
                    low=chart_df["Low"],
                    close=chart_df["Close"]
                )
            ]
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=650
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")