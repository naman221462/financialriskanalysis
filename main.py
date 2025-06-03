import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# ---------------------------
# Functions
# ---------------------------

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Handle multiple or single ticker cases
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']  # Get adjusted close prices when multiple tickers
    else:
        data = data[['Close']]
        data.columns = tickers  # Rename column to ticker name

    return data.dropna(axis=1, how='all')


def calculate_daily_returns(price_df):
    return price_df.pct_change(fill_method=None).dropna()


def calculate_var(returns, confidence=0.95):
    return returns.quantile(1 - confidence)


def calculate_sharpe(returns, risk_free_rate=0.01):
    excess_return = returns.mean() - (risk_free_rate / 252)
    return (excess_return / returns.std()) * np.sqrt(252)


def portfolio_return(weights, returns_df):
    weights_series = pd.Series(weights, index=returns_df.columns)
    return returns_df.dot(weights_series)


# ---------------------------
# Configuration
# ---------------------------
tickers = ['AAPL', 'MSFT', 'GOOG']
weights = np.array([0.4, 0.3, 0.3])  # Must match number of valid tickers
start_date = '2020-01-01'
end_date = '2024-12-31'

# ---------------------------
# Data Retrieval & Processing
# ---------------------------
prices = download_data(tickers, start_date, end_date)

# Adjust weights if some tickers failed
valid_tickers = prices.columns
if len(valid_tickers) != len(tickers):
    print(f"⚠️ Warning: Some tickers failed. Using available tickers: {list(valid_tickers)}")
    weights = weights[:len(valid_tickers)]
    weights = weights / weights.sum()  # Rebalance weights to sum to 1

returns = calculate_daily_returns(prices)
port_returns = portfolio_return(weights, returns)

# ---------------------------
# Risk Metrics Calculation
# ---------------------------
mean_return = port_returns.mean()
volatility = port_returns.std()
annual_volatility = volatility * np.sqrt(252)
var_95 = calculate_var(port_returns, 0.95)
sharpe = calculate_sharpe(port_returns)

# ---------------------------
# Plot Distribution
# ---------------------------
plt.figure(figsize=(10, 6))
sns.histplot(port_returns, bins=50, kde=True, color='skyblue')
plt.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.4f}')
plt.title('Portfolio Daily Return Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Print Summary
# ---------------------------
print("\n--- Portfolio Risk Metrics ---")
print(f"Assets Used: {list(valid_tickers)}")
print(f"Mean Daily Return: {mean_return:.4f}")
print(f"Daily Volatility: {volatility:.4f}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Value at Risk (95%): {var_95:.4f}")
print(f"Sharpe Ratio: {sharpe:.2f}")
