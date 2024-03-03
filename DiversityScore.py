import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from datetime import datetime

def read_portfolio_from_csv(file_path):
    """Read portfolio from a CSV file with 'Ticker' and 'Weight' columns."""
    try:
        portfolio_df = pd.read_csv(file_path, names=['Ticker', 'Weight'], header=None)
        portfolio = dict(zip(portfolio_df['Ticker'].str.upper(), portfolio_df['Weight']))
        if abs(sum(portfolio.values()) - 1) < 0.0001:
            return portfolio
        else:
            print("Total weight does not equal 1. Please check your portfolio." + str(sum(portfolio.values())))
            return None
    except Exception as e:
        print(f"Error reading the portfolio file: {e}")
        return None

def fetch_data(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Adj Close']
    sectors = {ticker: yf.Ticker(ticker).info['sector'] for ticker in tickers}
    return data, sectors

def calculate_sector_allocation(portfolio, sectors):
    sector_allocation = {sector: 0 for sector in set(sectors.values())}
    for ticker, weight in portfolio.items():
        sector = sectors.get(ticker, "Unknown")
        if sector in sector_allocation:
            sector_allocation[sector] += weight
    return sector_allocation
def highest_sector_allocation(sector_allocation):
    highest_sector = max(sector_allocation, key=sector_allocation.get)
    return highest_sector

def calculate_correlation(data):
    corr_matrix = data.pct_change().corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values,1)].mean()
    return corr_matrix, avg_corr

def recommend_sectors(sector_allocation):
    benchmark_sectors = {
        'Information Technology': 26.1, 'Health Care': 14.5, 'Financials': 12.9,
        'Consumer Discretionary': 9.9, 'Industrials': 8.6, 'Communication Services': 8.2,
        'Consumer Staples': 7.4, 'Energy': 4.5, 'Utilities': 2.9,
        'Materials': 2.6, 'Real Estate': 2.5
    }
    underrepresented_sectors = []
    for sector, benchmark_weight in benchmark_sectors.items():
        user_weight = sector_allocation.get(sector, 0)
        if user_weight < benchmark_weight:
            underrepresented_sectors.append(sector)
    return underrepresented_sectors

def main():
    file_path = input("Enter the file path of your portfolio CSV: ") #copy and paste the file path of your portfolio csv
    user_portfolio = read_portfolio_from_csv(file_path)
    if user_portfolio is None:
        return
    tickers = list(user_portfolio.keys())
    data, sectors = fetch_data(tickers)
    user_sector_allocation = calculate_sector_allocation(user_portfolio, sectors)
    _, avg_corr = calculate_correlation(data)
    underrepresented_sectors = recommend_sectors(user_sector_allocation)
    
    print(f"Recommended sectors to add for better diversification: {', '.join(underrepresented_sectors)}")
    print(f"Average correlation among assets: {avg_corr:.2f}")
    print(f"Most heavily weighted sector: {highest_sector_allocation(user_sector_allocation)}")

if __name__ == "__main__":
    main()
