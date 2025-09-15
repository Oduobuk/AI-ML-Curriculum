"""
Exercise 1: Solutions
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

# Solution 1: Create a MultiIndex DataFrame
df_multi = df1.set_index(['date', 'category'])
print("MultiIndex DataFrame:")
print(df_multi)

# Solution 2: Merge the two DataFrames
merged_df = pd.merge(df1, df2, on=['date', 'category'])
print("\nMerged DataFrame:")
print(merged_df)

# Solution 3: Create a pivot table
pivot_df = df1.pivot_table(
    values='value',
    index='date',
    columns='category',
    aggfunc='mean'
)
print("\nPivot Table:")
print(pivot_df)
