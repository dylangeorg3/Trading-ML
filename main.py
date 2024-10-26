
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

MODEL_PATH = 'models/trading_model.pkl'


# from functions import *


def load_data(ticker, period):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    return data


def process_data(data):
    data.ffill(inplace=True)  # Forward-fill missing values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns), scaler


def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data['MACD'] = compute_macd(data['Close'])

    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    return data


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')
    return accuracy

def backtest_strategy(predictions, data):
    pass

def simple_strategy(predictions, data):
    initial_cash = 10000  # Starting with $10,000
    holdings = 0
    cash = initial_cash

    for i in range(1, len(predictions)):
        # Only buy or sell if the 'Close' price is greater than 0
        if data['Close'][i] > 0:
            if predictions[i] == 1 and cash >= data['Close'][i]:  # Buy condition
                holdings += int(cash // data['Close'][i])  # Use int() for whole shares
                cash -= holdings * data['Close'][i]
            elif predictions[i] == -1 and holdings > 0:  # Sell condition
                cash += holdings * data['Close'][i]
                holdings = 0

    final_value = cash + holdings * data['Close'].iloc[-1] if data['Close'].iloc[-1] > 0 else cash
    percentage = (final_value - initial_cash) / initial_cash * 100
    return final_value - initial_cash, percentage  # Return total profit/loss

def main():
    period = '1y'
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'NFLX', 'PYPL', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'IBM', 'ADP', 'INTU']

    for ticker in stocks:
        # Load and preprocess data
        data = load_data(ticker, period)
        data, scaler = process_data(data)
        data = add_technical_indicators(data)

        # Define features and target
        X = data.drop(columns=['Target'])
        y = data['Target']

        # Train model
        model, X_test, y_test = train_model(X, y)
        save_model(model, MODEL_PATH)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Run backtesting strategy
        predictions = model.predict(X)
        profit_loss, percentage = simple_strategy(predictions, data)
        print(f"{ticker} Strategy profit/loss: ${profit_loss:.2f}")
        print(f"{percentage:.2f}% Proft/ Loss\n")





if __name__ == "__main__": 
    main()





