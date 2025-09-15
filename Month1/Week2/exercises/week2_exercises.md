# Week 2: Python for Data Science - Exercises

## Exercise 1: NumPy Fundamentals

### 1.1 Array Creation and Operations
Create a 3x3 NumPy array containing numbers from 1 to 9. Then:
1. Calculate the sum of all elements
2. Calculate the mean of each row
3. Find the maximum value in each column
4. Create a new array with elements squared

### 1.2 Array Indexing and Slicing
Given the following array:
```python
import numpy as np
data = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
```
1. Extract the second row
2. Extract the third column
3. Get the subarray consisting of the first two rows and last two columns
4. Create a boolean array where elements are greater than 5

## Exercise 2: Pandas DataFrames

### 2.1 DataFrame Creation and Basic Operations
Create a pandas DataFrame containing the following data about books:
- Title: ['Python Basics', 'Data Science 101', 'Machine Learning', 'Deep Learning']
- Author: ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Wilson']
- Year: [2019, 2020, 2021, 2021]
- Pages: [310, 450, 520, 480]

Then:
1. Display basic information about the DataFrame
2. Sort the DataFrame by year in descending order
3. Add a new column 'Pages per Year' that calculates pages divided by years since 2000
4. Find the book with the maximum number of pages

### 2.2 Data Filtering and Grouping
Using the same DataFrame:
1. Filter books published after 2019
2. Calculate the average number of pages
3. Group books by year and calculate the total pages for each year
4. Create a new column 'Length' that categorizes books as 'Short' (<400 pages), 'Medium' (400-500), or 'Long' (>500)

## Exercise 3: Data Visualization

### 3.1 Basic Plots with Matplotlib
Using the books DataFrame from Exercise 2:
1. Create a bar plot showing number of pages for each book
2. Create a scatter plot of year vs. pages
3. Customize the plots with appropriate titles, labels, and colors

### 3.2 Advanced Visualization with Seaborn
1. Create a boxplot showing the distribution of pages by publication year
2. Create a pairplot of all numerical columns
3. Add appropriate titles and labels to all plots

## Exercise 4: Real-World Data Analysis

### 4.1 Data Loading and Cleaning
1. Load the 'titanic' dataset from seaborn
2. Check for missing values and handle them appropriately
3. Convert categorical variables to appropriate types
4. Create a new feature 'FamilySize' by combining 'sibsp' and 'parch'

### 4.2 Exploratory Data Analysis
1. Calculate basic statistics for numerical columns
2. Create visualizations to explore relationships between survival and other variables
3. Identify any interesting patterns or insights in the data

## Exercise 5: Feature Engineering

### 5.1 Creating New Features
Using the titanic dataset:
1. Create a new feature 'Title' by extracting titles from the 'Name' column
2. Create age groups (e.g., child, adult, senior)
3. Create a new feature 'IsAlone' indicating passengers traveling alone

### 5.2 Data Transformation
1. Normalize the 'Fare' column
2. One-hot encode the 'Embarked' column
3. Create a correlation heatmap of all numerical features

## Submission Guidelines
1. Complete all exercises in a Jupyter Notebook
2. Include markdown cells to explain your approach and findings
3. Ensure your code is well-commented
4. Submit your notebook and any additional files by the due date

---
*Note: Solutions will be provided after the submission deadline.*
