# Week 2: Python for Data Science - Glossary

## Core Libraries

### NumPy
- **Array**: A grid of values, all of the same type, indexed by a tuple of non-negative integers
- **Vectorization**: The use of optimized, pre-compiled code to perform operations on entire arrays at once
- **Broadcasting**: NumPy's ability to perform operations on arrays of different shapes
- **ndarray**: N-dimensional array object, the primary data structure in NumPy

### Pandas
- **DataFrame**: A 2D labeled data structure with columns of potentially different types
- **Series**: A one-dimensional labeled array capable of holding any data type
- **Index**: The axis labeling information (row labels)
- **GroupBy**: A process involving splitting data into groups, applying a function, and combining results

### Matplotlib/Seaborn
- **Figure**: The top-level container for all plot elements
- **Axes**: The area on which data is plotted with x and y axis
- **Subplot**: A grid of plots within a single figure
- **Figure-level vs Axes-level functions**: Distinction between functions that create a new figure versus those that plot on existing axes

## Data Science Concepts

### Data Structures
- **Tidy Data**: A standard way of organizing data where each variable is a column and each observation is a row
- **Long vs Wide Format**: Different ways of structuring data for analysis
- **Categorical Data**: Data that can take on a limited number of possible values

### Data Manipulation
- **Pivoting**: Reshaping data from long to wide format
- **Melting**: Reshaping data from wide to long format
- **Merging/Joining**: Combining data from different sources based on common keys
- **Concatenation**: Stacking DataFrames vertically or horizontally

### Data Cleaning
- **Missing Data Handling**: Techniques like imputation or removal of missing values
- **Outlier Detection**: Identifying and handling anomalous data points
- **Data Type Conversion**: Changing data between different types (e.g., string to datetime)

## Common Terms
- **Vectorized Operations**: Operations that work on entire arrays rather than individual elements
- **Method Chaining**: Calling multiple methods in a single statement
- **Boolean Indexing**: Filtering data using boolean conditions
- **Lambda Functions**: Small anonymous functions defined with the `lambda` keyword
- **List/Dict Comprehensions**: Concise ways to create lists/dictionaries

## Performance Terms
- **Vectorization**: Using array operations instead of loops for better performance
- **Memory Usage**: The amount of RAM needed to store data structures
- **Time Complexity**: How the execution time increases with input size
- **Profiling**: Measuring the performance of code to identify bottlenecks

## Visualization Terms
- **Figure-level vs Axes-level**: Whether a plotting function creates a new figure or plots on existing axes
- **Aesthetic Mappings**: How data variables are mapped to visual properties
- **Faceting**: Creating multiple plots based on the values of one or more variables
- **Color Palettes**: Sets of colors used in visualizations

## Common Functions/Methods
- **NumPy**: `array()`, `arange()`, `linspace()`, `zeros()`, `ones()`, `reshape()`, `mean()`, `sum()`
- **Pandas**: `read_csv()`, `head()`, `info()`, `describe()`, `groupby()`, `merge()`, `pivot_table()`
- **Matplotlib**: `plot()`, `scatter()`, `bar()`, `hist()`, `xlabel()`, `ylabel()`, `title()`
- **Seaborn**: `scatterplot()`, `lineplot()`, `barplot()`, `boxplot()`, `heatmap()`, `pairplot()`

## Best Practices
- **DRY (Don't Repeat Yourself)**: Avoid code duplication
- **Vectorization**: Prefer array operations over loops when possible
- **Method Chaining**: Chain operations for more readable code
- **Documentation**: Use docstrings and comments to explain complex operations
- **Version Control**: Use Git to track changes to your code
