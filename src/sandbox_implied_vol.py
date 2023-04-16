import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download S&P500 data
symbol = "^GSPC"
sp500 = yf.Ticker(symbol)

# Get option chain
option_chain = sp500.option_chain()

# Get implied volatility surface
iv = option_chain.iv

# Drop rows with missing data
iv = iv.dropna()

# Reshape data to have strike price as columns and maturity as rows
iv = pd.pivot_table(iv, values='Implied Volatility', index=['Expiration', 'Type'], columns=['Strike'])

# Sort by maturity date
iv = iv.sort_index(level=0)

# Set up plot
fig, ax = plt.subplots(figsize=(12,6))

# Plot implied volatility surface
X = iv.columns.values
Y = (iv.index.levels[0].values - iv.index.levels[0].min()) / pd.Timedelta(1, unit='D')
Z = iv.values.T
ax.contourf(X, Y, Z, 30, cmap='viridis')

# Add labels and legend
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity (Days)')
ax.set_title('S&P500 Implied Volatility Surface')
plt.colorbar(ax=ax)

# Show plot
plt.show()
