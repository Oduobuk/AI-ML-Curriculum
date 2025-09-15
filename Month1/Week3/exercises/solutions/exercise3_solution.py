"""
Exercise 3: Solutions - Time Series Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = np.random.normal(100, 10, 100).cumsum()
df = pd.DataFrame({'date': dates, 'price': prices})
df.set_index('date', inplace=True)

# Solution 1: Resample to weekly frequency
weekly_mean = df.resample('W').mean()
print("Weekly Mean:")
print(weekly_mean.head())

# Solution 2: 7-day rolling average
df['7_day_avg'] = df['price'].rolling(window=7).mean()

# Solution 3: Monthly statistics
monthly_stats = df['price'].resample('M').agg(['min', 'max', 'mean'])
print("\nMonthly Statistics:")
print(monthly_stats)

# Solution 4: Handle missing values
# Create some missing values
df_missing = df.copy()
df_missing.iloc[10:15] = np.nan

# Forward fill missing values
df_filled = df_missing.fillna(method='ffill')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price'], 'b-', label='Original')
plt.plot(df_filled.index, df_filled['price'], 'r--', label='Forward Filled')
plt.title('Time Series with Missing Values Handled')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_series_handling.png')
plt.close()
