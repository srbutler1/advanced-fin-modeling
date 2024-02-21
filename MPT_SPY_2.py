import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

# Input tickers
tickers = input("Enter tickers separated by commas: ").split(',')
#enter risk free rate
rf = float(input("Enter the risk free rate: "))
# Add SPY for benchmark comparison but exclude it from the portfolio optimization
tickers.append('SPY')

# Get historical data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate expected returns and sample covariance for the selected tickers (excluding SPY)
mu = mean_historical_return(prices[tickers[:-1]])
S = sample_cov(prices[tickers[:-1]])

# Calculate annual return and volatility for SPY
spy_annual_return = mean_historical_return(prices[['SPY']])
spy_annual_volatility = np.sqrt(sample_cov(prices[['SPY']])).iloc[0,0]

# Number of portfolios to simulate
num_portfolios = 1000
all_weights = np.zeros((num_portfolios, len(prices.columns)-1))  # Excluding SPY
return_arr = np.zeros(num_portfolios)
volatility_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.array(np.random.random(len(prices.columns)-1))  # Excluding SPY
    weights = weights / np.sum(weights)
    
    all_weights[i, :] = weights
    
    # Expected return
    return_arr[i] = np.dot(weights, mu)
    
    # Expected volatility
    volatility_arr[i] = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    
    # Sharpe Ratio
    sharpe_arr[i] = return_arr[i]-rf / volatility_arr[i]
    

# Portfolio with maximum Sharpe Ratio excluding SPY
max_sr_ret = return_arr[sharpe_arr.argmax()]
max_sr_vol = volatility_arr[sharpe_arr.argmax()]

# Portfolio with minimum volatility excluding SPY
min_vol_ret = return_arr[volatility_arr.argmin()]
min_vol_vol = volatility_arr[volatility_arr.argmin()]
#show maximum sharpe ratio portfolio
max_sr_weights = all_weights[sharpe_arr.argmax()]
print('Maximum Sharpe Ratio Portfolio Weights:')
for i in range(len(tickers)-1):
    print(tickers[i], ':', round(max_sr_weights[i]*100, 2), '%')

#show maximum sharpe ratio portfolio return and volatility
print('Maximum Sharpe Ratio Portfolio Return:', round(max_sr_ret*100, 2), '%')
print('Maximum Sharpe Ratio Portfolio Volatility:', round(max_sr_vol*100, 2), '%')

#show minimum volatility portfolio
min_vol_weights = all_weights[volatility_arr.argmin()]
print('Minimum Volatility Portfolio Weights:')
for i in range(len(tickers)-1):
    print(tickers[i], ':', round(min_vol_weights[i]*100, 2), '%')
#show minimum volatility portfolio return and volatility
print('Minimum Volatility Portfolio Return:', round(min_vol_ret*100, 2), '%')
print('Minimum Volatility Portfolio Volatility:', round(min_vol_vol*100, 2), '%')

# Plotting the 1000 random portfolios along with SPY
plt.figure(figsize=(12,8))
plt.scatter(volatility_arr, return_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

#plot the tangent line to the efficient frontier
x = np.linspace(0, 0.3, 100)
y = max_sr_ret/max_sr_vol * x
plt.plot(x, y, 'r', label='Capital Market Line')


# Mark the portfolio with maximum Sharpe Ratio
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black', label='Maximum Sharpe Ratio')

# Mark the portfolio with minimum volatility
plt.scatter(min_vol_vol, min_vol_ret, c='blue', s=50, edgecolors='black', label='Minimum Volatility')

# Mark SPY
plt.scatter(spy_annual_volatility, spy_annual_return, c='orange', s=100, marker='*', label='SPY')

plt.legend(labelspacing=0.8)
plt.title('Portfolio Optimization with SPY Benchmark')
plt.show()



