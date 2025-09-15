"""
Exercise 2: Data Loading and Cleaning - Solutions
"""

import pandas as pd
import numpy as np
import json
from io import StringIO
import re

# Sample data
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

def load_and_clean_csv(data):
    """Load and clean CSV data."""
    # Load the CSV data
    df = pd.read_csv(StringIO(data))
    
    # Handle missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['email'] = df['email'].fillna('unknown@example.com')
    
    # Fill missing salaries with median of department
    df['salary'] = df.groupby('department')['salary']\
        .transform(lambda x: x.fillna(x.median()))
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Convert data types
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['department'] = df['department'].astype('category')
    
    # Handle salary outliers using IQR
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['salary'] = np.where(df['salary'] > upper_bound, upper_bound, df['salary'])
    
    return df

def load_excel_data():
    """Create and load sample Excel data."""
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
    
    # Load the Excel file
    loaded_df = pd.read_excel(excel_file, parse_dates=['last_restocked'])
    
    return loaded_df

def load_and_clean_json():
    """Load and clean JSON data."""
    # Sample JSON data
    json_data = """
    [
        {"user_id": 1, "name": "John Doe", "purchases": [{"item": "Laptop", "amount": 1200}, {"item": "Mouse", "amount": 50}]},
        {"user_id": 2, "name": "Jane Smith", "purchases": [{"item": "Phone", "amount": 800}]},
        {"user_id": 3, "name": "Bob Johnson", "purchases": [{"item": "Tablet", "amount": 350}, {"item": "Case", "amount": 25}, {"item": "Screen Protector", "amount": 15}]},
        {"user_id": 4, "name": "Alice Brown", "purchases": []}
    ]
    """
    
    # Parse JSON
    data = json.loads(json_data)
    
    # Normalize the nested data
    normalized_data = []
    for user in data:
        user_id = user['user_id']
        name = user['name']
        
        if user['purchases']:
            for purchase in user['purchases']:
                normalized_data.append({
                    'user_id': user_id,
                    'name': name,
                    'item': purchase['item'],
                    'amount': purchase['amount']
                })
        else:
            normalized_data.append({
                'user_id': user_id,
                'name': name,
                'item': None,
                'amount': None
            })
    
    return pd.DataFrame(normalized_data)

def validate_data(df):
    """Perform data validation on a DataFrame."""
    issues = {}
    clean_df = df.copy()
    
    # Check for missing values
    missing = clean_df.isnull().sum()
    if missing.any():
        issues['missing_values'] = missing[missing > 0].to_dict()
    
    # Check for duplicate rows
    duplicates = clean_df.duplicated().sum()
    if duplicates > 0:
        issues['duplicate_rows'] = f"{duplicates} duplicate rows found"
    
    # Validate email format if email column exists
    if 'email' in clean_df.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = ~clean_df['email'].str.match(email_pattern, na=False)
        if invalid_emails.any():
            issues['invalid_emails'] = clean_df.loc[invalid_emails, 'email'].tolist()
    
    # Check for outliers in numeric columns
    numeric_cols = clean_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in clean_df.columns:  # Check if column exists
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)
            if outliers.any():
                issues[f'outliers_in_{col}'] = {
                    'count': outliers.sum(),
                    'indices': clean_df[outliers].index.tolist()
                }
    
    # Additional custom validations
    if 'age' in clean_df.columns:
        invalid_ages = (clean_df['age'] < 18) | (clean_df['age'] > 100)
        if invalid_ages.any():
            issues['invalid_ages'] = clean_df.loc[invalid_ages, 'age'].to_dict()
    
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
    clean_df, issues = validate_data(df_cleaned)
    print("\nValidation issues:")
    for issue, details in issues.items():
        print(f"- {issue}: {details}")

if __name__ == "__main__":
    main()
