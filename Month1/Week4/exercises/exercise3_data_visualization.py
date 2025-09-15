"""
Exercise 3: Data Visualization with Matplotlib and Seaborn

This exercise covers:
- Basic plotting with Matplotlib
- Statistical visualizations with Seaborn
- Customizing plots
- Creating subplots and layouts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn')
sns.set_palette('colorblind')

# Sample data
def generate_sample_data():
    """Generate sample sales data for visualization."""
    np.random.seed(42)
    
    # Date range for one year
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Generate sales data with seasonality and trend
    trend = np.linspace(100, 200, len(dates))
    seasonality = 50 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 10, len(dates))
    sales = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int)
    })
    
    # Add some categorical data
    categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Other']
    df['category'] = np.random.choice(categories, size=len(df), p=[0.3, 0.2, 0.2, 0.2, 0.1])
    
    # Add customer segments
    segments = ['New', 'Returning', 'VIP']
    df['segment'] = np.random.choice(segments, size=len(df), p=[0.5, 0.4, 0.1])
    
    return df

# Exercise 1: Line plot with Matplotlib
def plot_sales_trend(df):
    """
    Create a line plot showing sales trend over time.
    
    Requirements:
    - Use Matplotlib
    - Add title and axis labels
    - Add grid lines
    - Customize line style and color
    - Add a rolling 7-day average line
    - Save the plot to a file
    """
    # TODO: Your code here
    pass

# Exercise 2: Bar plot with Seaborn
def plot_sales_by_category(df):
    """
    Create a bar plot showing total sales by category.
    
    Requirements:
    - Use Seaborn
    - Order categories by sales (highest to lowest)
    - Add value labels on top of bars
    - Customize colors
    - Add appropriate title and labels
    """
    # TODO: Your code here
    pass

# Exercise 3: Box plot and violin plot
def plot_sales_distribution(df):
    """
    Create side-by-side box plot and violin plot of sales by day of week.
    
    Requirements:
    - Create a figure with two subplots (1 row, 2 columns)
    - Left subplot: Box plot
    - Right subplot: Violin plot
    - Add appropriate titles and labels
    - Customize the appearance
    """
    # TODO: Your code here
    pass

# Exercise 4: Heatmap of sales by day of week and month
def plot_sales_heatmap(df):
    """
    Create a heatmap showing average sales by day of week and month.
    
    Requirements:
    - Pivot the data to create a matrix of day_of_week (rows) x month (columns)
    - Create a heatmap using Seaborn
    - Add a colorbar
    - Customize the colormap
    - Add appropriate title and labels
    """
    # TODO: Your code here
    pass

# Exercise 5: Advanced visualization
def plot_customer_segment_analysis(df):
    """
    Create a multi-panel visualization showing customer segment analysis.
    
    Requirements:
    - Create a 2x2 grid of subplots
    - Top-left: Pie chart of customer segments
    - Top-right: Box plot of sales by segment
    - Bottom-left: Stacked bar chart of category distribution by segment
    - Bottom-right: Line plot of monthly sales trend by segment
    - Add appropriate titles and labels
    - Customize the appearance
    """
    # TODO: Your code here
    pass

def main():
    # Generate sample data
    df = generate_sample_data()
    
    print("=== Exercise 1: Sales Trend Line Plot ===")
    plot_sales_trend(df)
    
    print("\n=== Exercise 2: Sales by Category Bar Plot ===")
    plot_sales_by_category(df)
    
    print("\n=== Exercise 3: Sales Distribution by Day of Week ===")
    plot_sales_distribution(df)
    
    print("\n=== Exercise 4: Sales Heatmap ===")
    plot_sales_heatmap(df)
    
    print("\n=== Exercise 5: Customer Segment Analysis ===")
    plot_customer_segment_analysis(df)
    
    plt.show()

if __name__ == "__main__":
    main()
