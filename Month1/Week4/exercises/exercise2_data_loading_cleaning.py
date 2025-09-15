"""
Exercise 2: Data Loading and Cleaning

This exercise covers:
- Loading data from various sources (CSV, Excel, JSON, SQL)
- Handling missing values
- Removing duplicates
- Data type conversion
- Handling outliers
"""

import pandas as pd
import numpy as np
import json
from io import StringIO

# Sample data for exercises
csv_data = """id,name,age,email,salary,join_date,department
1,John Doe,32,john@example.com,85000,2020-05-15,Engineering
2,Jane Smith,28,jane@example.com,78000,2021-02-20,Marketing
3,Bob Johnson,,bob@example.com,92000,2019-11-10,Engineering
4,Alice Brown,35,alice@example.com,,2022-01-05,Sales
5,Charlie Wilson,41,charlie@example.com,110000,2018-07-22,Marketing
6,Jane Smith,28,jane@example.com,78000,2021-02-20,Marketing
7,David Lee,29,david@example.com,95000,2021-08-14,Engineering
8,Eva Green,33,,88000,2020-09-30,Sales
9,Frank White,45,frank@example.com,125000,2017-12-01,Executive
10,Grace Hall,31,grace@example.com,89000,2021-03-18,HR
"""

# Exercise 1: Load and clean CSV data
def load_and_clean_csv(data):
    """
    Load CSV data from a string and perform basic cleaning:
    1. Handle missing values
    2. Remove duplicates
    3. Convert data types
    4. Handle outliers in salary
    
    Returns:
        Cleaned DataFrame
    """
    # TODO: Load the CSV data from the string
    # df = pd.read_csv(...)
    
    # TODO: Handle missing values
    # - Fill missing ages with the median age
    # - Fill missing emails with 'unknown@example.com'
    # - Fill missing salaries with the median salary for the department
    
    # TODO: Remove duplicate rows
    
    # TODO: Convert data types
    # - join_date to datetime
    # - department to category
    
    # TODO: Handle salary outliers using IQR method
    # Cap salaries at Q3 + 1.5*IQR
    
    return df

# Exercise 2: Load data from Excel
def load_excel_data():
    """
    Create a sample Excel file and load it with pandas.
    """
    # Create sample data
    data = {
        'product_id': [101, 102, 103, 104, 105],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
        'price': [1200, 45.99, 89.99, 349.99, 129.99],
        'stock': [15, 100, 75, 25, 50],
        'last_restocked': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-03-15', '2023-02-28']
    }
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    excel_file = 'products.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Products', index=False)
    
    # TODO: Load the Excel file and return the DataFrame
    # Make sure to parse the 'last_restocked' column as datetime
    
    return loaded_df

# Exercise 3: Load and clean JSON data
def load_and_clean_json():
    """
    Load and clean JSON data.
    """
    # Sample JSON data
    json_data = """
    [
        {"user_id": 1, "name": "John Doe", "purchases": [{"item": "Laptop", "amount": 1200}, {"item": "Mouse", "amount": 50}]},
        {"user_id": 2, "name": "Jane Smith", "purchases": [{"item": "Phone", "amount": 800}]},
        {"user_id": 3, "name": "Bob Johnson", "purchases": [{"item": "Tablet", "amount": 350}, {"item": "Case", "amount": 25}, {"item": "Screen Protector", "amount": 15}]},
        {"user_id": 4, "name": "Alice Brown", "purchases": []}
    ]
    """
    
    # TODO: Load the JSON data into a DataFrame
    # df = pd.read_json(...)
    
    # TODO: Normalize the nested 'purchases' data and create a flat table
    # with one row per purchase, including user details
    
    return normalized_df

# Exercise 4: Data validation
def validate_data(df):
    """
    Perform data validation on a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (clean_df, issues) where issues is a dictionary of problems found
    """
    issues = {}
    clean_df = df.copy()
    
    # TODO: Check for missing values
    
    # TODO: Check for duplicate rows
    
    # TODO: Validate data types
    
    # TODO: Check for outliers in numeric columns
    
    # TODO: Add custom validation rules
    # Example: Check if email addresses are valid
    
    return clean_df, issues

def main():
    print("=== Exercise 1: Load and clean CSV data ===")
    df_cleaned = load_and_clean_csv(csv_data)
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    print("\nData types:")
    print(df_cleaned.dtypes)
    
    print("\n=== Exercise 2: Load Excel data ===")
    excel_df = load_excel_data()
    print("\nExcel data:")
    print(excel_df)
    print("\nData types:")
    print(excel_df.dtypes)
    
    print("\n=== Exercise 3: Load and clean JSON data ===")
    json_df = load_and_clean_json()
    print("\nNormalized JSON data:")
    print(json_df)
    
    print("\n=== Exercise 4: Data validation ===")
    # Use the cleaned CSV data for validation
    clean_df, issues = validate_data(df_cleaned)
    print("\nValidation issues:")
    for issue, details in issues.items():
        print(f"- {issue}: {details}")

if __name__ == "__main__":
    main()
