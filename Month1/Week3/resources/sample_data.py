"""
Sample datasets for Week 3 exercises
"""

import pandas as pd
import numpy as np

def generate_sales_data():
    """Generate sample sales data with datetime index"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    data = {
        'sales': np.random.poisson(100, size=len(dates)) * 
                (1 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)) +
                np.random.normal(0, 10, size=len(dates)),
        'visitors': np.random.poisson(50, size=len(dates)) * 
                   (1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)) +
                   np.random.normal(0, 5, size=len(dates)),
        'promotion': np.random.choice([0, 1], size=len(dates), p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add weekly seasonality
    df['sales'] = df['sales'] * (1 + 0.2 * np.sin(2 * np.pi * df.index.dayofweek / 7))
    
    # Add some missing values
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'sales'] = np.nan
    
    return df

def generate_customer_data():
    """Generate sample customer data"""
    np.random.seed(42)
    
    categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Other']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'customer_id': range(1, 1001),
        'age': np.random.normal(35, 10, 1000).astype(int).clip(18, 80),
        'income': np.random.lognormal(10, 0.4, 1000).astype(int),
        'category': np.random.choice(categories, 1000, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'region': np.random.choice(regions, 1000),
        'loyalty_score': np.random.beta(2, 5, 1000) * 100
    }
    
    return pd.DataFrame(data)

def generate_stock_data():
    """Generate sample stock price data"""
    np.random.seed(42)
    
    # Generate random walk for stock prices
    n_days = 252  # Trading days in a year
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create date range (business days only)
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.lognormal(8, 1, n_days).astype(int)
    }, index=dates)
    
    # Add some features
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
    
    return df

if __name__ == "__main__":
    # Generate and save sample datasets
    sales_df = generate_sales_data()
    customer_df = generate_customer_data()
    stock_df = generate_stock_data()
    
    # Save to CSV
    sales_df.to_csv('resources/sales_data.csv')
    customer_df.to_csv('resources/customer_data.csv')
    stock_df.to_csv('resources/stock_data.csv')
    
    print("Sample datasets generated and saved to resources/")
    print(f"- Sales data: {len(sales_df)} days")
    print(f"- Customer data: {len(customer_df)} customers")
    print(f"- Stock data: {len(stock_df)} trading days")
