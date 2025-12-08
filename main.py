from fastapi import FastAPI
import yfinance as yf
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Stock Prediction API Running"}

@app.get("/predict")
def predict(symbol: str):
    try:
        data = yf.download(symbol, period="10d", interval="1d", progress=False)

        if data.empty:
            return {"error": "Invalid stock symbol"}

        close = data["Close"]
        last_price = float(close.iloc[-1])
        
        support = float(close.tail(20).min())
        resistance = float(close.tail(20).max())

        trend = "UP" if last_price > close.mean() else "DOWN"

        target_up = last_price * 1.03
        target_down = last_price * 0.97

        return {
            "symbol": symbol,
            "last_price": last_price,
            "trend": trend,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "target_up": round(target_up, 2),
            "target_down": round(target_down, 2)
        }

    except Exception as e:
        return {"error": str(e)}
