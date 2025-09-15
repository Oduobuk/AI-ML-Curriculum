"""
Exercise 3: Time Series Analysis

This exercise covers:
- Working with datetime objects
- Time-based indexing
- Resampling and window operations
- Handling missing data in time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = np.random.normal(100, 10, 100).cumsum()
df = pd.DataFrame({'date': dates, 'price': prices})
df.set_index('date', inplace=True)

# Exercise 1: Resampling
# Your task: Resample the data to weekly frequency and calculate the mean price for each week
# Expected: A Series with weekly mean prices
# Your code here


# Exercise 2: Rolling windows
# Your task: Calculate a 7-day rolling average of the price
# Expected: A Series with the rolling average values
# Your code here


# Exercise 3: Time-based operations
# Your task: Calculate the day-over-day percentage change in price
# Expected: A Series with percentage changes
# Your code here


# Exercise 4: Handling missing data
# Create some missing values
df_missing = df.copy()
df_missing.iloc[10:15] = np.nan

# Your task: Handle the missing values using the following methods:
# 1. Forward fill
# 2. Linear interpolation
# 3. Rolling window imputation (7-day window)
# Expected: Three different Series with different imputation methods
# Your code here


# Exercise 5: Time series decomposition
# Your task: Decompose the time series into trend, seasonal, and residual components
# Use statsmodels.tsa.seasonal.seasonal_decompose
# Expected: A decomposition object with trend, seasonal, and residual components
# Your code here


# Exercise 6: Visualization
# Your task: Create a plot with the following elements:
# 1. Original price series
# 2. 7-day rolling average
# 3. Highlight periods with missing values
# Expected: A single plot with all elements clearly labeled
# Your code here
