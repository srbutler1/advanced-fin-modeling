import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_fred_series(series_id, start_date):
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': 'cb4bfc8fd985c90a14e45b242eff77ce',
        'file_type': 'json',
        'start_date': start_date,
    }
    response = requests.get(url, params=params)
    data = response.json()['observations']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

# Fetch Federal Funds Rate starting from a specific date
start_date = '2000-01-01'  # Adjust as needed
ffr_df = fetch_fred_series('DFF', start_date)

# Filter for the first day of each month
ffr_df_first_of_month = ffr_df[ffr_df['date'].dt.is_month_start]

# Fetch Consumer Spending (Personal Consumption Expenditures)
pce_df = fetch_fred_series('PCE', start_date)

# Filter for the first day of each month for consistency
pce_df_first_of_month = pce_df[pce_df['date'].dt.is_month_start]

# Display the first few rows of each DataFrame as a check
print("Federal Funds Rate (First of Each Month):")
print(ffr_df_first_of_month.tail(120))
print("\nConsumer Spending (PCE) (First of Each Month):")
print(pce_df_first_of_month.tail(120))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Federal Funds Rate', color=color)
ax1.plot(ffr_df_first_of_month['date'], ffr_df_first_of_month['value'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Consumer Spending (PCE)', color=color)
ax2.plot(pce_df_first_of_month['date'], pce_df_first_of_month['value'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ten_years_ago = datetime.now() - pd.DateOffset(years=10)
ax1.set_xlim([ten_years_ago, datetime.now()])

plt.title('Federal Funds Rate vs. Consumer Spending (Last 10 Years)')
fig.tight_layout()  # Adjust layout to not clip labels
plt.show()










