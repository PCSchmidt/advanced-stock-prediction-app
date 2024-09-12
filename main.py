# Import necessary libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
import logging
import traceback
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pmdarima, use a fallback if not available
try:
    from pmdarima import auto_arima
    USE_PMDARIMA = True
except ImportError:
    logger.warning("pmdarima not found. Using a simple parameter search for ARIMA and SARIMA models.")
    USE_PMDARIMA = False

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize a simple in-memory cache
cache: Dict[Tuple[str, str, str, str], Tuple[List[float], List[float], datetime]] = {}

def prepare_data(data: pd.Series, n_steps: int = 60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(data: pd.Series) -> pd.Series:
    # Remove any NaN values
    data = data.dropna()
    
    # Check for stationarity and difference if necessary
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data)
    if result[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
        data = data.diff().dropna()
    
    return data

def simple_parameter_search(data, seasonal=False):
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                if seasonal:
                    for P in range(0, 2):
                        for D in range(0, 2):
                            for Q in range(0, 2):
                                try:
                                    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, 12))
                                    results = model.fit()
                                    aic = results.aic
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, 12)
                                except:
                                    continue
                else:
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
    
    return best_order, best_seasonal_order

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Stock Price Prediction API"}

@app.get("/predict/{symbol}")
async def predict_stock(symbol: str, start_date: str, end_date: str, model: str):
    try:
        cache_key = (symbol, start_date, end_date, model)
        if cache_key in cache:
            historical, predictions, timestamp = cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return {
                    "symbol": symbol,
                    "historical": historical,
                    "predictions": predictions,
                    "cached": True
                }

        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}' in the given date range")
        
        data = hist['Close']
        
        if model == "arima":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    processed_data = preprocess_data(data)
                    if USE_PMDARIMA:
                        auto_model = auto_arima(processed_data, start_p=1, start_q=1, max_p=5, max_q=5, m=1, d=None, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
                        order = auto_model.order
                    else:
                        order, _ = simple_parameter_search(processed_data)
                    logger.info(f"Best ARIMA order: {order}")
                    model_fit = ARIMA(processed_data, order=order).fit()
                    forecast = model_fit.forecast(steps=30)
                    # If we differenced the data, we need to add back the last value
                    if not np.array_equal(data, processed_data):
                        forecast = forecast + data.iloc[-1]
                except Exception as e:
                    logger.error(f"Error fitting ARIMA model: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"Error fitting ARIMA model: {str(e)}")
        elif model == "sarima":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    processed_data = preprocess_data(data)
                    if USE_PMDARIMA:
                        auto_model = auto_arima(processed_data, start_p=1, start_q=1, max_p=5, max_q=5, m=12, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
                        order = auto_model.order
                        seasonal_order = auto_model.seasonal_order
                    else:
                        order, seasonal_order = simple_parameter_search(processed_data, seasonal=True)
                    logger.info(f"Best SARIMA order: {order}, seasonal_order: {seasonal_order}")
                    model_fit = SARIMAX(processed_data, order=order, seasonal_order=seasonal_order).fit()
                    forecast = model_fit.forecast(steps=30)
                    # If we differenced the data, we need to add back the last value
                    if not np.array_equal(data, processed_data):
                        forecast = forecast + data.iloc[-1]
                except Exception as e:
                    logger.error(f"Error fitting SARIMA model: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"Error fitting SARIMA model: {str(e)}")
        elif model == "garch":
            returns = 100 * data.pct_change().dropna()
            model_fit = arch_model(returns, vol="Garch", p=1, q=1).fit(disp='off')
            forecast = model_fit.forecast(horizon=30)
            if hasattr(forecast, 'mean') and hasattr(forecast.mean, 'iloc'):
                forecast_means = forecast.mean.iloc[-1].values
                last_price = data.iloc[-1]
                forecast = last_price * (1 + forecast_means / 100)
            else:
                raise ValueError("Unexpected forecast structure from GARCH model")
        elif model == "lstm":
            try:
                logger.info("Starting LSTM model preparation")
                X, y, scaler = prepare_data(data)
                logger.info(f"LSTM input shape: {X.shape}, output shape: {y.shape}")
                
                lstm_model = create_lstm_model((X.shape[1], 1))
                logger.info("LSTM model created")
                
                logger.info("Starting LSTM model training")
                lstm_model.fit(X, y, epochs=100, batch_size=32, verbose=0)
                logger.info("LSTM model training completed")
                
                last_sequence = X[-1]
                forecast = []
                logger.info("Starting LSTM prediction")
                for _ in range(30):
                    next_pred = lstm_model.predict(last_sequence.reshape(1, X.shape[1], 1))
                    forecast.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1)
                    last_sequence[-1] = next_pred
                logger.info("LSTM prediction completed")
                
                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
                logger.info("LSTM forecast inverse transformed")
            except Exception as e:
                logger.error(f"Error in LSTM model: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                logger.error(f"System info: {sys.version}")
                raise HTTPException(status_code=500, detail=f"Error in LSTM model: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model '{model}' specified. Available models are: arima, sarima, garch, lstm")
        
        historical = data.tolist()
        predictions = forecast.tolist() if isinstance(forecast, np.ndarray) else forecast.tolist()

        cache[cache_key] = (historical, predictions, datetime.now())
        
        return {
            "symbol": symbol,
            "historical": historical,
            "predictions": predictions,
            "cached": False
        }
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        if "No data found" in str(e):
            raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}' in the given date range")
        elif "Invalid input" in str(e):
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)