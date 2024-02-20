import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf

# Override pandas_datareader's get_data_yahoo method to use yfinance directly
yf.pdr_override()

def fetch_data(ticker, market_index, start_date, end_date):
    try:
        stock_data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        market_data = pdr.get_data_yahoo(market_index, start=start_date, end=end_date)
        return stock_data, market_data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None, None

def calculate_daily_returns(data):
    return data['Adj Close'].pct_change()

def calculate_beta(stock_returns, market_returns):
    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

def calculate_capm_return(beta, risk_free_rate, market_return):
    return risk_free_rate + beta * (market_return - risk_free_rate)

def plot_data(stock_returns, market_returns, ticker, beta, capm_return):
    plt.figure(figsize=(10, 6))
    plt.scatter(market_returns, stock_returns, alpha=0.5)
    plt.title(f"{ticker} Returns vs. Market Returns")
    plt.xlabel("Market Returns")
    plt.ylabel(f"{ticker} Stock Returns")
    plt.grid(True)
    
    # Annotate beta and CAPM return on the plot
    plt.annotate(f'Beta: {beta:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
    plt.annotate(f'Expected CAPM Return: {capm_return:.2%}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, verticalalignment='top')

    plt.savefig(f"{ticker}_returns_plot.png")
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    market_index = 'SPY'  # Using SPY directly
    risk_free_rate = float(input("Enter the risk-free rate (as a decimal): "))
    start_date = dt.datetime.now() - dt.timedelta(days=365 * 5)  # 5 years back from today
    end_date = dt.datetime.now()

    stock_data, market_data = fetch_data(ticker, market_index, start_date, end_date)
    if stock_data is not None and market_data is not None:
        stock_returns = calculate_daily_returns(stock_data).dropna()
        market_returns = calculate_daily_returns(market_data).dropna()

        # Ensure both returns are aligned in length before calculating beta
        min_length = min(len(stock_returns), len(market_returns))
        stock_returns = stock_returns.iloc[:min_length]
        market_returns = market_returns.iloc[:min_length]

        beta = calculate_beta(stock_returns.values, market_returns.values)
        avg_market_return = market_returns.mean() * 252  # Annualizing the return
        capm_return = calculate_capm_return(beta, risk_free_rate, avg_market_return)

        plot_data(stock_returns, market_returns, ticker, beta, capm_return)

        print(f"Ticker: {ticker}")
        print(f"Beta: {beta:.2f}")
        print(f"Expected CAPM Return: {capm_return:.2%}")
    else:
        print("Failed to fetch data for the given ticker or market index.")
