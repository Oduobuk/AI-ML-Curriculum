"""
Exercise 2: Data Visualization

This exercise covers:
- Advanced Matplotlib customization
- Seaborn visualizations
- Interactive plots with Plotly
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

# Sample data
tips = sns.load_dataset('tips')

# Exercise 1: Create a customized Matplotlib plot
# Your task: Create a scatter plot of total_bill vs tip with the following customizations:
# - Different colors for each day
# - Add title and axis labels
# - Add a legend
# - Set figure size to (10, 6)
# Your code here


# Exercise 2: Create a Seaborn visualization
# Your task: Create a box plot showing distribution of total_bill by day and time
# - Use hue for 'time' (lunch/dinner)
# - Add appropriate title and labels
# - Set figure size to (12, 6)
# Your code here


# Exercise 3: Create an interactive Plotly visualization
# Your task: Create an interactive scatter plot with the following features:
# - x-axis: total_bill
# - y-axis: tip
# - Color points by 'smoker'
# - Size points by 'size' of the party
# - Add hover information showing 'day' and 'time'
# - Add a title
# Your code here


# Exercise 4: Advanced customization
# Your task: Create a figure with 2 subplots:
# - Left: A histogram of 'total_bill' with kernel density estimate
# - Right: A violin plot of 'total_bill' by 'day'
# - Add appropriate titles and labels
# Your code here


# Exercise 5: Time series visualization
# Your task: Create a time series line plot with the following data:
# - Generate a date range for 30 days
# - Create a Series with random walk data (cumulative sum of normal random numbers)
# - Add proper labels and title
# - Add a horizontal line at y=0
# Your code here
