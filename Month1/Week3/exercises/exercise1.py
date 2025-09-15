"""
Exercise 1: Advanced Pandas Operations

This exercise covers advanced operations in Pandas including:
- Multi-indexing
- Advanced merging
- Pivot tables
"""

import pandas as pd
import numpy as np

# Sample data for exercises
data1 = {
    'date': pd.date_range('2023-01-01', periods=6, freq='D'),
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'value': [10, 20, 30, 40, 50, 60]
}

df1 = pd.DataFrame(data1)

data2 = {
    'date': pd.date_range('2023-01-01', periods=6, freq='D'),
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'price': [100, 200, 150, 250, 120, 180]
}

df2 = pd.DataFrame(data2)

# Exercise 1: Create a MultiIndex DataFrame
# Your code here

# Exercise 2: Merge the two DataFrames on 'date' and 'category'
# Your code here

# Exercise 3: Create a pivot table showing the mean 'value' by 'category' and 'date'
# Your code here
