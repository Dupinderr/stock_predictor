# # 📈 Indian Stock Recommendation System

An AI-powered machine learning system that analyzes historical Indian stock market data and generates **BUY / HOLD / SELL** recommendations for Nifty 50 stocks using technical indicators and an XGBoost classification model.

---

## 🚀 Project Overview

The Indian Stock Recommendation System is an end-to-end machine learning project that:

- Collects historical stock market data for Nifty 50 stocks
- Performs preprocessing and feature engineering
- Computes technical indicators
- Trains an XGBoost classification model
- Predicts stock recommendations
- Provides an interactive Streamlit dashboard for real-time analysis

This project demonstrates practical implementation of:

- Financial data analysis
- Machine learning for market prediction
- Feature engineering
- Model deployment
- Interactive web app development

---

## 🎯 Objective

To build an intelligent recommendation engine capable of predicting:

- **BUY**
- **HOLD**
- **SELL**

based on historical stock behavior and technical indicators.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- XGBoost
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Financial Data
- yFinance
- TA (Technical Analysis library)

### Visualization
- Plotly
- Matplotlib

### Web Application
- Streamlit

---

## 📂 Project Structure

```bash
stock_market_ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── scraper.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   └── xgboost_stock_model.pkl
│
└── modeltesting.ipynb
