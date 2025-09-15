"""
Exercise 3: Time Series Analysis

This exercise covers:
- Working with datetime objects
- Time-based indexing
- Resampling and window operations
"""

import pandas as pd
import numpy as np

# Generate sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = np.random.normal(100, 10, 100).cumsum()
df = pd.DataFrame({'date': dates, 'price': prices})
df.set_index('date', inplace=True)

# Exercise 1: Resample the data to weekly frequency and calculate the mean
# Your code here

# Exercise 2: Calculate a 7-day rolling average
# Your code here

# Exercise 3: Resample to monthly frequency and show min, max, and mean prices
# Your code here

# Exercise 4: Handle missing values using forward fill
# First, create some missing values
df_missing = df.copy()
df_missing.iloc[10:15] = np.nan
# Your code to fill missing values here
