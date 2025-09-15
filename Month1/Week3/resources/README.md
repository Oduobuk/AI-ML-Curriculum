# Week 3 Resources

This directory contains resources for Week 3: Advanced Data Analysis and Visualization.

## File Structure

```
resources/
├── cheatsheets/               # Quick reference guides
│   ├── pandas_cheatsheet.md     # Pandas operations reference
│   ├── matplotlib_seaborn_cheatsheet.md  # Visualization reference
│   └── time_series_cheatsheet.md # Time series analysis reference
├── sample_data.py             # Script to generate sample datasets
├── sales_data.csv             # Sample sales time series data
├── customer_data.csv          # Sample customer data
└── stock_data.csv             # Sample stock price data
```

## Cheatsheets

1. **Pandas Cheatsheet**
   - Core data structures
   - Data selection and filtering
   - Data cleaning and manipulation
   - Grouping and aggregation
   - Time series operations

2. **Matplotlib & Seaborn Cheatsheet**
   - Basic plotting with Matplotlib
   - Advanced visualization with Seaborn
   - Customization and styling
   - Subplots and figures
   - Saving and exporting plots

3. **Time Series Cheatsheet**
   - Time series manipulation
   - Resampling and frequency conversion
   - Rolling windows and time-based operations
   - Time series decomposition
   - Stationarity and differencing

## Sample Datasets

### Sales Data (`sales_data.csv`)
- **Description**: Daily sales data with visitor counts and promotion flags
- **Features**:
  - `sales`: Daily sales amount
  - `visitors`: Daily number of visitors
  - `promotion`: Binary flag for promotion days
- **Time Period**: Full year 2023
- **Use Cases**:
  - Time series analysis
  - Impact of promotions
  - Daily/Weekly seasonality

### Customer Data (`customer_data.csv`)
- **Description**: Customer demographic and preference data
- **Features**:
  - `customer_id`: Unique identifier
  - `age`: Customer age
  - `income`: Annual income
  - `category`: Preferred product category
  - `region`: Geographic region
  - `loyalty_score`: Customer loyalty metric (0-100)
- **Use Cases**:
  - Customer segmentation
  - Demographic analysis
  - Feature engineering

### Stock Data (`stock_data.csv`)
- **Description**: Daily stock price and volume data
- **Features**:
  - `price`: Daily closing price
  - `volume`: Trading volume
  - `returns`: Daily returns
  - `volatility`: 20-day rolling volatility
- **Time Period**: 252 trading days
- **Use Cases**:
  - Financial time series analysis
  - Volatility modeling
  - Technical indicators

## Usage

To use the sample data in your Python code:

```python
import pandas as pd

# Load the data
sales = pd.read_csv('resources/sales_data.csv', index_col=0, parse_dates=True)
customers = pd.read_csv('resources/customer_data.csv')
stocks = pd.read_csv('resources/stock_data.csv', index_col=0, parse_dates=True)
```

## Regenerating Sample Data

To regenerate the sample datasets:

```bash
# Install required packages
pip install pandas numpy

# Run the data generation script
python resources/sample_data.py
```

## Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Examples](https://seaborn.pydata.org/examples/index.html)
- [Pandas Time Series/Date functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
