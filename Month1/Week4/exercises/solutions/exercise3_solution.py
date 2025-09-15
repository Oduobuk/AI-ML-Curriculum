"""
Exercise 3: Data Visualization with Matplotlib and Seaborn - Solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style
plt.style.use('seaborn')
sns.set_palette('colorblind')

# Sample data generation
def generate_sample_data():
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

def plot_sales_trend(df):
    """Create a line plot showing sales trend over time with 7-day rolling average."""
    plt.figure(figsize=(14, 7))
    
    # Plot daily sales
    plt.plot(df['date'], df['sales'], 
             color='#3498db', 
             alpha=0.5, 
             label='Daily Sales')
    
    # Calculate and plot 7-day rolling average
    df['rolling_7d'] = df['sales'].rolling(window=7).mean()
    plt.plot(df['date'], df['rolling_7d'], 
             color='#e74c3c', 
             linewidth=2, 
             label='7-Day Moving Average')
    
    # Customize the plot
    plt.title('Daily Sales Trend with 7-Day Moving Average', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Sales ($)', fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format y-axis as currency
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sales_trend.png', dpi=300, bbox_inches='tight')
    print("Sales trend plot saved as 'sales_trend.png'")

def plot_sales_by_category(df):
    """Create a bar plot showing total sales by category."""
    # Calculate total sales by category
    sales_by_category = df.groupby('category')['sales'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    ax = sns.barplot(x=sales_by_category.index, 
                    y=sales_by_category.values,
                    palette='viridis')
    
    # Add value labels on top of bars
    for i, v in enumerate(sales_by_category.values):
        ax.text(i, v + 1000, f'${v:,.0f}', 
                ha='center', 
                va='bottom',
                fontweight='bold')
    
    # Customize the plot
    plt.title('Total Sales by Category', fontsize=16, pad=20)
    plt.xlabel('Category', fontsize=12, labelpad=10)
    plt.ylabel('Total Sales ($)', fontsize=12, labelpad=10)
    
    # Format y-axis as currency
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sales_by_category.png', dpi=300, bbox_inches='tight')
    print("Sales by category plot saved as 'sales_by_category.png'")

def plot_sales_distribution(df):
    """Create side-by-side box plot and violin plot of sales by day of week."""
    # Map day numbers to names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['day_of_week'].map(dict(enumerate(day_names)))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    sns.boxplot(x='day_name', y='sales', data=df, ax=ax1, order=day_names)
    ax1.set_title('Box Plot of Sales by Day of Week', fontsize=14)
    ax1.set_xlabel('Day of Week', fontsize=12)
    ax1.set_ylabel('Sales ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(x='day_name', y='sales', data=df, ax=ax2, order=day_names)
    ax2.set_title('Distribution of Sales by Day of Week', fontsize=14)
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Sales ($)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Format y-axis as currency
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sales_distribution.png', dpi=300, bbox_inches='tight')
    print("Sales distribution plot saved as 'sales_distribution.png'")

def plot_sales_heatmap(df):
    """Create a heatmap showing average sales by day of week and month."""
    # Create pivot table
    pivot_table = df.pivot_table(
        values='sales',
        index='day_name',
        columns='month',
        aggfunc='mean'
    )
    
    # Reorder index to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(day_order)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table, 
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=.5,
        cbar_kws={'label': 'Average Sales ($)'}
    )
    
    # Customize the plot
    plt.title('Average Sales by Day of Week and Month', fontsize=16, pad=20)
    plt.xlabel('Month', fontsize=12, labelpad=10)
    plt.ylabel('Day of Week', fontsize=12, labelpad=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sales_heatmap.png', dpi=300, bbox_inches='tight')
    print("Sales heatmap saved as 'sales_heatmap.png'")

def plot_customer_segment_analysis(df):
    """Create a multi-panel visualization for customer segment analysis."""
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 16))
    
    # 1. Pie chart of customer segments (top-left)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    segment_counts = df['segment'].value_counts()
    ax1.pie(
        segment_counts,
        labels=segment_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel'),
        wedgeprops=dict(edgecolor='white')
    )
    ax1.set_title('Customer Segments', fontsize=14)
    
    # 2. Box plot of sales by segment (top-right)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sns.boxplot(
        x='segment',
        y='sales',
        data=df,
        order=['New', 'Returning', 'VIP'],
        ax=ax2
    )
    ax2.set_title('Sales Distribution by Segment', fontsize=14)
    ax2.set_xlabel('Customer Segment', fontsize=12)
    ax2.set_ylabel('Sales ($)', fontsize=12)
    
    # 3. Stacked bar chart of category distribution by segment (bottom-left)
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    category_segment = pd.crosstab(
        index=df['segment'],
        columns=df['category'],
        normalize='index'
    )
    category_segment.plot(
        kind='bar',
        stacked=True,
        ax=ax3,
        colormap='viridis'
    )
    ax3.set_title('Category Distribution by Segment', fontsize=14)
    ax3.set_xlabel('Customer Segment', fontsize=12)
    ax3.set_ylabel('Proportion', fontsize=12)
    ax3.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Line plot of monthly sales trend by segment (bottom-right)
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    
    # Resample to monthly sales by segment
    monthly_sales = df.groupby([pd.Grouper(key='date', freq='M'), 'segment'])['sales'].sum().reset_index()
    
    sns.lineplot(
        x='date',
        y='sales',
        hue='segment',
        data=monthly_sales,
        marker='o',
        ax=ax4
    )
    
    ax4.set_title('Monthly Sales Trend by Segment', fontsize=14)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Total Sales ($)', fontsize=12)
    
    # Format y-axis as currency
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax4.yaxis.set_major_formatter(formatter)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add main title
    plt.suptitle('Customer Segment Analysis', fontsize=18, y=1.02)
    
    # Save the plot
    plt.savefig('customer_segment_analysis.png', dpi=300, bbox_inches='tight')
    print("Customer segment analysis plot saved as 'customer_segment_analysis.png'")

def main():
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    
    print("\n=== Exercise 1: Sales Trend Line Plot ===")
    plot_sales_trend(df)
    
    print("\n=== Exercise 2: Sales by Category Bar Plot ===")
    plot_sales_by_category(df)
    
    print("\n=== Exercise 3: Sales Distribution by Day of Week ===")
    plot_sales_distribution(df)
    
    print("\n=== Exercise 4: Sales Heatmap ===")
    plot_sales_heatmap(df)
    
    print("\n=== Exercise 5: Customer Segment Analysis ===")
    plot_customer_segment_analysis(df)
    
    print("\nAll visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main()
