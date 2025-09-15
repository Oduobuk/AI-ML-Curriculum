# Time Series Analysis Cheat Sheet

## 1. Core Time Series Operations

### Creating Time Series Data
```python
import pandas as pd
import numpy as np

# Create date range
dates = pd.date_range('2023-01-01', periods=7, freq='D')

# Create time series data
ts = pd.Series(np.random.randn(7), index=dates)

# Create DataFrame with datetime index
df = pd.DataFrame({
    'value': np.random.randn(100),
    'date': pd.date_range('2023-01-01', periods=100, freq='D')
}).set_index('date')
```

### Time-based Indexing
```python
# Select by label
ts['2023-01-03']

# Select by partial string
df['2023-01']  # All of January 2023
df['2023-01-10':'2023-01-20']  # Date range

# First/last n periods
df.first('5D')  # First 5 days
df.last('2W')   # Last 2 weeks
```

### Shifting and Lagging
```python
# Shift data forward/backward
df.shift(1)      # Shift forward 1 period
df.shift(-1)     # Shift backward 1 period

# Calculate percentage change
df.pct_change()  # Percentage change between current and prior element

# Difference between consecutive elements
df.diff()        # First discrete difference
```

## 2. Resampling and Frequency Conversion

### Downsampling (to lower frequency)
```python
# Resample to weekly frequency
weekly = df.resample('W').mean()

# Multiple aggregations
weekly_stats = df.resample('W').agg(['mean', 'min', 'max', 'std'])

# Custom resampling function
def custom_resampler(array):
    return array[-1]  # Return last value in period

custom = df.resample('W').apply(custom_resampler)
```

### Upsampling (to higher frequency)
```python
# Upsample to hourly frequency
hourly = df.resample('H').asfreq()  # Creates NaNs

# Forward fill
ffilled = df.resample('H').ffill()

# Backward fill
bfilled = df.resample('H').bfill()

# Interpolation
interpolated = df.resample('H').interpolate(method='linear')
```

### Rolling Windows
```python
# Simple moving average
rolling_mean = df.rolling(window=7).mean()

# Expanding window
cumulative_mean = df.expanding().mean()

# Exponentially weighted moving average
ewma = df.ewm(span=30, adjust=False).mean()

# Custom rolling function
def custom_roll(x):
    return np.percentile(x, 75)

rolling_custom = df.rolling(30).apply(custom_roll)
```

## 3. Time Zone Handling

```python
# Localize to timezone
tz_naive = pd.date_range('2023-01-01', periods=3, freq='H')
tz_aware = tz_naive.tz_localize('US/Eastern')

# Convert between timezones
tz_utc = tz_aware.tz_convert('UTC')

# Handle DST transitions
# Note: Use 'infer_dst' for ambiguous times during DST transitions
dst_dates = pd.date_range('2023-03-12', periods=5, freq='H', tz='US/Eastern')
```

## 4. Time Series Decomposition

### Additive Model
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
result_add = seasonal_decompose(series, model='additive', period=7)
result_add.plot()
```

### Multiplicative Model
```python
# Multiplicative decomposition
result_mul = seasonal_decompose(series, model='multiplicative', period=7)
result_mul.plot()
```

### STL Decomposition
```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust)
stl = STL(series, period=7, robust=True)
result_stl = stl.fit()
result_stl.plot()
```

## 5. Handling Missing Data

### Detection
```python
# Check for missing values
df.isnull().sum()

# Find indices of missing values
missing_indices = df[df['value'].isnull()].index
```

### Imputation Methods
```python
# Forward fill
df_ffill = df.ffill()

# Backward fill
df_bfill = df.bfill()

# Linear interpolation
df_linear = df.interpolate(method='linear')

# Time-based interpolation
df_time = df.interpolate(method='time')

# Using rolling mean
df_rolling = df.fillna(df.rolling(3, min_periods=1).mean())
```

## 6. Time Series Visualization

### Basic Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
df.plot(figsize=(12, 6))
plt.title('Time Series Plot')
plt.ylabel('Value')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# Multiple time series
df[['A', 'B']].plot(subplots=True, layout=(2, 1), figsize=(12, 8))
```

### Rolling Statistics
```python
# Plot rolling mean and std
rolling = df.rolling(window=30)
df_mean = rolling.mean()
df_std = rolling.std()

plt.figure(figsize=(12, 6))
plt.plot(df, label='Original')
plt.plot(df_mean, label='Rolling Mean')
plt.fill_between(df_std.index, 
                (df_mean - 2*df_std).iloc[:,0], 
                (df_mean + 2*df_std).iloc[:,0], 
                color='b', alpha=0.2)
plt.legend()
plt.show()
```

### Seasonal Decomposition Plot
```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df, model='additive', period=7)
result.plot()
plt.tight_layout()
plt.show()
```

## 7. Time Series Stationarity

### Testing for Stationarity
```python
from statsmodels.tsa.stattools import adfuller, kpss

# Augmented Dickey-Fuller test
result = adfuller(df['value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value:.3f}')

# KPSS test
result = kpss(df['value'])
print(f'KPSS Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[3].items():
    print(f'   {key}: {value:.3f}')
```

### Making a Series Stationary
```python
# Differencing
df_diff = df.diff().dropna()

# Log transformation
df_log = np.log(df)

# Seasonal differencing
df_seasonal_diff = df.diff(periods=7).dropna()  # For weekly seasonality
```

## 8. Feature Engineering

### Time-based Features
```python
# Extract date components
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = df.index.dayofweek >= 5

# Business day features
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())
df['is_holiday'] = df.index.isin(holidays)
```

### Lag Features
```python
# Create lag features
for i in range(1, 4):
    df[f'lag_{i}'] = df['value'].shift(i)
```

### Rolling Statistics
```python
# Rolling mean
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()

# Rolling standard deviation
df['rolling_std_7'] = df['value'].rolling(window=7).std()

# Rolling quantile
df['rolling_q25_7'] = df['value'].rolling(window=7).quantile(0.25)
```

## 9. Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Create time series cross-validation object
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train and evaluate model
```

## 10. Common Time Series Models

### ARIMA
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=7)
```

### Prophet
```python
from prophet import Prophet

# Prepare data
df_prophet = df.reset_index()
df_prophet = df_prophet.rename(columns={'date': 'ds', 'value': 'y'})

# Fit model
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=30)

# Make predictions
forecast = model.predict(future)
model.plot(forecast)
```

## 11. Performance Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
```

## 12. Useful Pandas Functions

### Date Ranges
```python
# Business days
bday = pd.offsets.BDay()
pd.date_range('2023-01-01', periods=5, freq=bday)

# Month ends
pd.date_range('2023-01-01', periods=12, freq='M')

# Custom business days
from pandas.tseries.offsets import CustomBusinessDay
weekmask = 'Mon Tue Wed Thu Fri'
holidays = ['2023-01-01', '2023-07-04']
bday_custom = CustomBusinessDay(weekmask=weekmask, holidays=holidays)
pd.date_range('2023-01-01', periods=10, freq=bday_custom)
```

### Time Deltas
```python
# Create time delta
delta = pd.Timedelta(days=5, hours=3)

# Add to datetime
new_date = pd.Timestamp('2023-01-01') + delta

# Business day offsets
from pandas.tseries.offsets import BDay
new_date = pd.Timestamp('2023-01-01') + 5 * BDay()
```

## 13. Working with Time Zones

```python
# Localize to timezone
tz_naive = pd.Timestamp('2023-01-01 12:00')
tz_aware = tz_naive.tz_localize('US/Eastern')

# Convert between timezones
tz_utc = tz_aware.tz_convert('UTC')

# Handle ambiguous times (during DST transitions)
tz_ambiguous = pd.Timestamp('2023-11-05 01:30:00', tz='US/Eastern')

# Handle non-existent times (spring forward)
tz_nonexistent = pd.Timestamp('2023-03-12 02:30:00', tz='US/Eastern')
```

## 14. Performance Optimization

### Vectorized Operations
```python
# Avoid loops, use vectorized operations
# Slow:
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * 2

# Fast:
df['new_col'] = df['col1'] * 2
```

### Efficient Data Types
```python
# Convert to appropriate dtypes
df['small_int'] = df['small_int'].astype('int8')
df['category'] = df['category'].astype('category')

# Downcast numeric columns
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
```

### Chunk Processing
```python
# Process large files in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    process(chunk)
```

## 15. Common Pitfalls

1. **Time Zone Naivety**: Always be aware of time zones when working with timestamps.
2. **Missing Data**: Handle missing values appropriately for your analysis.
3. **Data Leakage**: Be careful not to use future data when creating features.
4. **Stationarity**: Many time series models assume stationarity.
5. **Seasonality**: Account for seasonal patterns in your analysis.
6. **Performance**: Be mindful of performance with large time series data.
7. **Data Alignment**: Ensure proper alignment when working with multiple time series.
8. **Frequency**: Be consistent with time series frequency.
9. **Edge Cases**: Handle edge cases like DST transitions and leap years.
10. **Documentation**: Document your assumptions and preprocessing steps.
