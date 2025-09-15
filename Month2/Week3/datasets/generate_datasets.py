"""
Dataset generation script for Week 3: Decision Trees and Ensemble Methods
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def create_classification_dataset(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42):
    """Create a synthetic classification dataset with mixed feature types."""
    np.random.seed(random_state)
    
    # Create numerical features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=1,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=[0.8] + [0.2/(n_classes-1)] * (n_classes-1) if n_classes > 1 else [1.0],
        flip_y=0.05,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Create column names
    num_cols = [f'num_{i+1}' for i in range(n_features)]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=num_cols)
    
    # Add some categorical features
    n_cat = min(5, n_features // 4)  # Add some categorical features
    cat_cols = []
    
    for i in range(n_cat):
        n_categories = np.random.randint(2, 6)  # 2-5 categories
        cat_name = f'cat_{i+1}'
        cat_cols.append(cat_name)
        df[cat_name] = np.random.choice([f'cat_{i}_val_{j}' for j in range(n_categories)], 
                                      size=n_samples)
    
    # Add target column
    df['target'] = y
    
    # Add some missing values (5%)
    for col in df.columns[:-1]:  # Don't add missing values to target
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['target'], random_state=random_state
    )
    
    return train_df, test_df, num_cols, cat_cols

def create_regression_dataset(n_samples=1000, n_features=20, n_informative=10, random_state=42):
    """Create a synthetic regression dataset with non-linear relationships."""
    np.random.seed(random_state)
    
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=10.0,
        random_state=random_state
    )
    
    # Create non-linear relationships
    for i in range(min(5, n_informative)):
        y += 5 * np.sin(X[:, i] * 2)  # Add sine waves
        y += 3 * (X[:, i] ** 2)  # Add quadratic terms
    
    # Add some outliers
    outlier_mask = np.random.random(n_samples) < 0.01  # 1% outliers
    y[outlier_mask] += 100 * np.random.randn(np.sum(outlier_mask))
    
    # Create column names
    feature_cols = [f'feature_{i+1}' for i in range(n_features)]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    
    # Add some categorical features
    n_cat = min(3, n_features // 5)  # Add some categorical features
    cat_cols = []
    
    for i in range(n_cat):
        n_categories = np.random.randint(2, 5)  # 2-4 categories
        cat_name = f'cat_{i+1}'
        cat_cols.append(cat_name)
        df[cat_name] = np.random.choice([f'cat_{i}_val_{j}' for j in range(n_categories)], 
                                      size=n_samples)
    
    # Add target column
    df['target'] = y
    
    # Add some missing values (5%)
    for col in df.columns[:-1]:  # Don't add missing values to target
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=random_state
    )
    
    return train_df, test_df, feature_cols, cat_cols

def create_time_series_dataset(n_samples=1000, n_features=10, n_informative=5, 
                             freq='D', start_date='2020-01-01', random_state=42):
    """Create a time series dataset for regression."""
    np.random.seed(random_state)
    
    # Generate base features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=5.0,
        random_state=random_state
    )
    
    # Create time index
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Create column names
    feature_cols = [f'feature_{i+1}' for i in range(n_features)]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_cols, index=dates)
    
    # Add time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Add some seasonality
    df['seasonal_1'] = 5 * np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['seasonal_2'] = 3 * np.cos(2 * np.pi * df.index.hour / 24) if 'H' in freq else 0
    
    # Update feature columns
    feature_cols.extend(['day_of_week', 'month', 'quarter', 'seasonal_1'])
    if 'H' in freq:
        feature_cols.append('seasonal_2')
    
    # Add target with some lag features
    noise = np.random.normal(0, 2, n_samples)
    df['target'] = (
        0.5 * X[:, 0] + 
        0.3 * X[:, 1] ** 2 + 
        2 * np.sin(X[:, 2] * 2) + 
        0.8 * X[:, 3] * X[:, 4] +
        df['seasonal_1'] +
        noise
    )
    
    # Add some missing values (5%)
    for col in df.columns[:-1]:  # Don't add missing values to target
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Split into train and test (time-based split)
    train_size = int(0.8 * n_samples)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df, feature_cols, ['day_of_week', 'month', 'quarter']

def save_datasets():
    """Generate and save all datasets."""
    # Create output directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # 1. Binary Classification
    print("Generating binary classification dataset...")
    train_clf, test_clf, num_cols, cat_cols = create_classification_dataset(
        n_samples=10000, n_features=30, n_informative=15, n_classes=2, random_state=42
    )
    
    # Save with metadata
    train_clf.to_csv('datasets/classification_train.csv', index=False)
    test_clf.to_csv('datasets/classification_test.csv', index=False)
    
    with open('datasets/classification_metadata.txt', 'w') as f:
        f.write(f"Numerical columns: {', '.join(num_cols)}\n")
        f.write(f"Categorical columns: {', '.join(cat_cols)}\n")
    
    # 2. Multi-class Classification
    print("Generating multi-class classification dataset...")
    train_multi, test_multi, num_cols, cat_cols = create_classification_dataset(
        n_samples=10000, n_features=25, n_informative=10, n_classes=5, random_state=43
    )
    
    train_multi.to_csv('datasets/multiclass_train.csv', index=False)
    test_multi.to_csv('datasets/multiclass_test.csv', index=False)
    
    with open('datasets/multiclass_metadata.txt', 'w') as f:
        f.write(f"Numerical columns: {', '.join(num_cols)}\n")
        f.write(f"Categorical columns: {', '.join(cat_cols)}\n")
    
    # 3. Regression
    print("Generating regression dataset...")
    train_reg, test_reg, num_cols, cat_cols = create_regression_dataset(
        n_samples=10000, n_features=20, n_informative=12, random_state=44
    )
    
    train_reg.to_csv('datasets/regression_train.csv', index=False)
    test_reg.to_csv('datasets/regression_test.csv', index=False)
    
    with open('datasets/regression_metadata.txt', 'w') as f:
        f.write(f"Numerical columns: {', '.join(num_cols)}\n")
        f.write(f"Categorical columns: {', '.join(cat_cols)}\n")
    
    # 4. Time Series
    print("Generating time series dataset...")
    train_ts, test_ts, num_cols, cat_cols = create_time_series_dataset(
        n_samples=2000, n_features=15, n_informative=8, 
        freq='D', start_date='2020-01-01', random_state=45
    )
    
    train_ts.to_csv('datasets/timeseries_train.csv')
    test_ts.to_csv('datasets/timeseries_test.csv')
    
    with open('datasets/timeseries_metadata.txt', 'w') as f:
        f.write(f"Numerical columns: {', '.join(num_cols)}\n")
        f.write(f"Categorical columns: {', '.join(cat_cols)}\n")
    
    print("All datasets generated successfully!")

if __name__ == "__main__":
    save_datasets()
