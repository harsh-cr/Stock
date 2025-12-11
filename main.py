from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model.pkl (trained on Kaggle)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def prepare_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    df = df.dropna()
    return df

@app.get("/")
def home():
    return {"status": "ML API working"}

@app.get("/predict")
def predict(symbol: str):
    try:
        data = yf.download(symbol, period="6mo", interval="1d")

        if data is None or len(data) == 0:
            return {"error": "Invalid symbol or no data found"}

        df = data.copy()
        df = prepare_features(df)

        if df.empty:
            return {"error": "Not enough data to compute features"}

        last_row = df.iloc[-1]

        X = np.array([[
            last_row["Return"],
            last_row["MA5"],
            last_row["MA20"],
            last_row["Volatility"]
        ]])

        prediction = model.predict(X)[0]

        return {
            "symbol": symbol,
            "predicted_price": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}
