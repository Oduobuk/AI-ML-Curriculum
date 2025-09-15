# Pandas Cheat Sheet

## Data Structures

### Series
```python
import pandas as pd

# Create Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Date range
s = pd.Series(range(6), index=pd.date_range('20230101', periods=6))
```

### DataFrame
```python
# Create DataFrame
df = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20230101'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
})
```

## Viewing Data

```python
df.head()       # First 5 rows
df.tail(3)      # Last 3 rows
df.index        # Row index
df.columns      # Column names
df.describe()   # Summary statistics
df.T            # Transpose
df.sort_index(axis=1, ascending=False)  # Sort by column names
df.sort_values(by='B')  # Sort by values in column B
```

## Selection

### Getting
```python
df['A']         # Column selection (returns Series)
df[0:3]         # Row selection by position
df['20230102':'20230104']  # Row selection by label
```

### Selection by Label
```python
df.loc[dates[0]]            # Select row by label
df.loc[:, ['A', 'B']]      # Select columns A and B
df.loc['20230102':'20230104', ['A', 'B']]  # Select rows and columns
```

### Selection by Position
```python
df.iloc[3]          # Select row by integer position
df.iloc[3:5, 0:2]   # Rows 3-4, columns 0-1
df.iloc[[1, 2, 4], [0, 2]]  # Specific rows and columns
df.iloc[1:3, :]     # Rows 1-2, all columns
df.iloc[:, 1:3]     # All rows, columns 1-2
df.iloc[1, 1]       # Single value by position
```

## Boolean Indexing
```python
df[df.A > 0]                     # Rows where A > 0
df[df > 0]                       # Values > 0 (else NaN)
df[df['E'].isin(['one', 'two'])] # Filter by list of values
```

## Missing Data
```python
df.dropna(how='any')     # Drop rows with any NaN
df.fillna(value=5)       # Fill NaN with value
df.fillna(method='ffill')  # Forward fill
pd.isna(df)             # Boolean mask of NaN values
```

## Operations

### Stats
```python
df.mean()               # Mean of each column
df.mean(1)              # Mean of each row
df.apply(lambda x: x.max() - x.min())  # Apply function to columns
```

### String Methods
```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()  # Convert to lowercase
```

## Merge/Join/Concatenate
```python
# Concatenate
pd.concat([df1, df2, df3], axis=0)  # Stack vertically

# Merge (SQL-style)
pd.merge(left, right, on='key')

# Join
df1.join(df2)  # Join on index
```

## Grouping
```python
df.groupby('A').sum()           # Group by column A and sum
df.groupby(['A', 'B']).mean()   # Group by multiple columns
```

## Reshaping
```python
# Pivot
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

# Stack/Unstack
stacked = df2.stack()
stacked.unstack(1)
```

## Time Series
```python
# Date range
dates = pd.date_range('20230101', periods=6)

# Resampling
df.resample('M').mean()  # Monthly resampling

# Timezones
df.tz_localize('UTC').tz_convert('US/Eastern')
```

## Plotting
```python
df.plot()
df.plot.scatter(x='A', y='B')
```

## Input/Output

### CSV
```python
df.to_csv('file.csv')
pd.read_csv('file.csv')
```

### Excel
```python
df.to_excel('file.xlsx', sheet_name='Sheet1')
pd.read_excel('file.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

### HDF5
```python
df.to_hdf('file.h5', 'df')
pd.read_hdf('file.h5', 'df')
```

### SQL
```python
from sqlalchemy import create_engine
engine = create_engine('sqlite:///mydb.sqlite')
df.to_sql('table_name', engine)
pd.read_sql('SELECT * FROM table_name', engine)
```
