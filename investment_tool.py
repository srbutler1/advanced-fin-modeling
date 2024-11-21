#### for innputs H is for holdings of the top 500 funds and B is for the most bought stocks in the last quarter 
#### strategies are E for equal weight, S for sharpe ratio maximization and R for risk parity - or minimum volatility. Max sharpe has a min max of 1000 and 15000 respectively to keep diversity
#### the data being used is for historical data to calculate the sharpe ratio 
#### we could also include a portfolio that is a mirror of the mutual fund holdings 
import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

import logging

# Configure logging
logging.basicConfig(
    filename='debug_wrds_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Connect to WRDS
def connect_to_wrds(username):
    try:
        conn = wrds.Connection(wrds_username=username)
        print("Connected to WRDS!")
        logging.info("Connected to WRDS successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        logging.error(f"Error connecting to WRDS: {e}")
        return None


# Fetch the top 500 mutual funds by cumulative return over 10 years
def get_top_500_funds(conn):
    try:
        query = """
            WITH returns_10y AS (
                SELECT 
                    crsp_fundno,
                    EXP(SUM(LOG(1 + mret))) - 1 AS cumulative_return
                FROM 
                    crsp_q_mutualfunds.monthly_tna_ret_nav
                WHERE 
                    caldt >= CURRENT_DATE - INTERVAL '10 years'
                    AND mret IS NOT NULL
                GROUP BY 
                    crsp_fundno
            )
            SELECT 
                crsp_fundno, cumulative_return
            FROM 
                returns_10y
            ORDER BY 
                cumulative_return DESC
            LIMIT 500;
        """
        logging.info("Executing top 500 mutual funds query.")
        result = conn.raw_sql(query)
        if result.empty:
            logging.warning("Top 500 funds query returned no results.")
            print("Top 500 funds query returned no results.")
            return None
        print("Top 500 Funds by Cumulative Return:")
        print(result.head())
        logging.info(f"Top 500 funds query result:\n{result.head()}")
        return result['crsp_fundno'].tolist()
    except Exception as e:
        print(f"Error retrieving top 500 funds: {e}")
        logging.error(f"Error retrieving top 500 funds: {e}")
        return None


# Fetch tickers and associated values based on top funds
def fetch_tickers(conn, query_type="holdings", top_n=50, top_funds=None):
    if not top_funds:
        print("No top funds provided. Exiting.")
        logging.error("No top funds provided for ticker fetching.")
        return None, None

    try:
        if query_type == "holdings":
            query = f"""
                SELECT 
                    hc.ticker,
                    SUM(h.market_val) AS total_mutual_fund_holding_value
                FROM 
                    crsp.holdings AS h
                JOIN 
                    crsp.holdings_co_info AS hc
                ON 
                    h.crsp_company_key = hc.crsp_company_key
                JOIN 
                    crsp_q_mutualfunds.portnomap AS pm
                ON 
                    h.crsp_portno = pm.crsp_portno
                WHERE 
                    h.report_dt = (SELECT MAX(report_dt) FROM crsp.holdings)
                    AND pm.crsp_fundno IN ({','.join(map(str, top_funds))})
                GROUP BY 
                    hc.ticker
                ORDER BY 
                    total_mutual_fund_holding_value DESC
                LIMIT {top_n};
            """
        elif query_type == "bought":
            latest_two_dates_query = """
                SELECT DISTINCT report_dt
                FROM crsp.holdings
                ORDER BY report_dt DESC
                LIMIT 2;
            """
            logging.info("Fetching latest two report dates.")
            dates = conn.raw_sql(latest_two_dates_query)

            if len(dates) < 2:
                print("Not enough quarters of data available.")
                logging.warning("Not enough quarters of data available.")
                return None, None

            latest_date = dates['report_dt'].iloc[0]
            previous_date = dates['report_dt'].iloc[1]
            logging.info(f"Latest Date: {latest_date}, Previous Date: {previous_date}")

            query = f"""
                SELECT 
                    h1.ticker,
                    h1.security_name,
                    (SUM(h2.market_val) - SUM(h1.market_val)) AS change_in_market_value
                FROM 
                    crsp.holdings AS h1
                JOIN 
                    crsp.holdings AS h2
                ON 
                    h1.ticker = h2.ticker
                    AND h1.security_name = h2.security_name
                JOIN 
                    crsp_q_mutualfunds.portnomap AS pm
                ON 
                    h1.crsp_portno = pm.crsp_portno
                    AND h2.crsp_portno = pm.crsp_portno
                WHERE 
                    h1.report_dt = '{previous_date}'
                    AND h2.report_dt = '{latest_date}'
                    AND pm.crsp_fundno IN ({','.join(map(str, top_funds))})
                GROUP BY 
                    h1.ticker, h1.security_name
                ORDER BY 
                    change_in_market_value DESC
                LIMIT {top_n};
            """
        else:
            raise ValueError("Invalid query type. Use 'holdings' or 'bought'.")

        logging.info(f"Executing {query_type} query.")
        result = conn.raw_sql(query)
        if result.empty:
            print("No data returned for the selected query.")
            logging.warning(f"No data returned for the {query_type} query.")
            return None, None

        logging.info(f"{query_type.capitalize()} Query Result:\n{result.head()}")
        tickers = [ticker for ticker in result['ticker'] if ticker is not None]
        return tickers, result

    except Exception as e:
        print(f"Error fetching tickers: {e}")
        logging.error(f"Error fetching tickers: {e}")
        return None, None



# Optimize portfolio using MPT
def optimize_portfolio(tickers, rf, investment_amount, hist_data, strategy):
    try:
        # Set historical data range
        start_date = (datetime.today() - timedelta(days=int(hist_data) * 365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].dropna(axis=1)

        dropped_tickers = set(tickers) - set(prices.columns)
        if dropped_tickers:
            print(f"Tickers with insufficient data dropped: {', '.join(dropped_tickers)}")

        mu = mean_historical_return(prices)
        S = sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        if strategy == "E":
            weights = {ticker: 1 / len(tickers) for ticker in tickers}
        elif strategy == "S":
            ef.add_constraint(lambda w: w >= 0.01)  # Minimum weight
            ef.add_constraint(lambda w: w <= 0.15)  # Maximum weight
            weights = ef.max_sharpe(risk_free_rate=rf)
        elif strategy == "R":
            ef.add_objective(objective_functions.L2_reg, gamma=1)
            weights = ef.min_volatility()
        else:
            print("Invalid strategy. Using Sharpe-maximization by default.")
            weights = ef.max_sharpe(risk_free_rate=rf)

        cleaned_weights = ef.clean_weights()
        allocations = {ticker: weight * investment_amount for ticker, weight in cleaned_weights.items()}
        performance = ef.portfolio_performance(risk_free_rate=rf)

        # Generate Efficient Frontier plot
        plot_efficient_frontier(mu, S, rf, ef)
        return allocations, performance
    except Exception as e:
        print(f"Error during portfolio optimization: {e}")
        return None, None


def plot_efficient_frontier(mu, S, rf, ef):
    n_samples = 5000
    np.random.seed(42)
    w = np.random.dirichlet(np.ones(len(mu)), size=n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpe_ratios = (rets - rf) / stds

    plt.figure(figsize=(12, 8))
    plt.scatter(stds, rets, c=sharpe_ratios, cmap="viridis")
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Efficient Frontier with Capital Market Line")

    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = ef.portfolio_performance(risk_free_rate=rf)
    plt.scatter(max_sharpe_vol, max_sharpe_ret, c="red", s=100, label="Maximum Sharpe Ratio")

    x = np.linspace(0, max_sharpe_vol, 100)
    y = rf + max_sharpe_sr * x
    plt.plot(x, y, color="orange", label="Capital Market Line")

    plt.legend()
    plt.grid()
    plt.show()


# Main script
if __name__ == "__main__":
    username = input("Enter your WRDS username: ")
    conn = connect_to_wrds(username)

    if conn:
        top_funds = get_top_500_funds(conn)
        if not top_funds:
            print("Failed to retrieve top 500 funds. Exiting.")
            exit()

        list_choice = input("Choose Top 50 Holdings (H) or Most Bought Stocks (B): ").strip().upper()
        query_type = "holdings" if list_choice == "H" else "bought"

        tickers, result = fetch_tickers(conn, query_type=query_type, top_funds=top_funds)

        if not tickers:
            print("No valid tickers found. Exiting.")
            exit()

        print("\nTickers fetched:")
        print(result)
        if query_type == "bought":
            result = result.sort_values(by="change_in_market_value", ascending=False)
            plt.figure(figsize=(12, 6))
            plt.bar(result['ticker'], result['change_in_market_value'])
            plt.title("Most Bought Stocks by Change in Market Value")
            plt.xlabel("Stock Ticker")
            plt.ylabel("Change in Market Value (USD)")
        else:
            # Filter out rows with missing or invalid data
            result = result.dropna(subset=['ticker', 'total_mutual_fund_holding_value'])
            result = result.sort_values(by="total_mutual_fund_holding_value", ascending=False)
            
            # Ensure valid data types
            result['ticker'] = result['ticker'].astype(str)
            result['total_mutual_fund_holding_value'] = result['total_mutual_fund_holding_value'].astype(float)
            
            plt.figure(figsize=(12, 6))
            plt.bar(result['ticker'], result['total_mutual_fund_holding_value'])
            plt.title("Top 50 Holdings by Mutual Fund Holding Value")
            plt.xlabel("Stock Ticker")
            plt.ylabel("Holding Value (USD)")

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

        investment_amount = float(input("\nEnter the amount to invest: "))
        rf = float(input("Enter the risk-free rate (e.g., 0.03): "))
        hist_data = input("Enter years of historical data (1, 3, 5, 10): ").strip()
        strategy = input("Choose strategy: Equal-Weight (E), Sharpe-Max (S), Risk-Parity (R): ").strip().upper()

        allocations, performance = optimize_portfolio(tickers, rf, investment_amount, hist_data, strategy)

        if allocations:
            print("\nPortfolio Allocations:")
            for ticker, alloc in allocations.items():
                print(f"{ticker}: ${alloc:.2f}")

            print("\nPortfolio Performance:")
            print(f"Expected Return: {performance[0] * 100:.2f}%")
            print(f"Volatility: {performance[1] * 100:.2f}%")
            print(f"Sharpe Ratio: {performance[2]:.2f}")
