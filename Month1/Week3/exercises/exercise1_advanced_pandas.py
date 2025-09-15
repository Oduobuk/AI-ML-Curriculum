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
# Your task: Convert df1 to have a MultiIndex with 'date' and 'category' as indices
# Expected: A DataFrame with a two-level index
# Your code here


# Exercise 2: Merge the two DataFrames on 'date' and 'category'
# Your task: Perform an inner join on df1 and df2 using both 'date' and 'category' as keys
# Expected: A single DataFrame with columns from both df1 and df2
# Your code here


# Exercise 3: Create a pivot table showing the mean 'value' by 'category' and 'date'
# Your task: Use the pivot_table function to reshape the data
# Expected: A pivot table with dates as rows, categories as columns, and mean values as cells
# Your code here


# Exercise 4: Advanced grouping
# Your task: Group by 'category' and calculate multiple aggregations
# Expected: A DataFrame showing min, max, and mean of 'value' for each category
# Your code here


# Exercise 5: Time-based operations
# Your task: Set 'date' as index and calculate a 2-day rolling average of 'value'
# Expected: A Series with the rolling average values
# Your code here
