"""
Sample Data Generation for Linear Regression Exercises
"""
import numpy as np
import pandas as pd
import os

def generate_linear_data(n_samples=100, noise=1.0, random_state=42):
    """Generate linear data with noise."""
    np.random.seed(random_state)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * noise
    return X, y

def generate_housing_data(n_samples=1000, random_state=42):
    """Generate synthetic housing price data."""
    np.random.seed(random_state)
    
    # Generate features
    size = 800 + 2500 * np.random.rand(n_samples)  # House size in sq ft
    bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
    age = np.random.randint(0, 50, n_samples)      # House age in years
    
    # Generate target (price in $1000s)
    price = (
        50 * size / 1000 +  # Base price per sq ft
        20 * bedrooms ** 2 +  # Extra for more bedrooms
        -0.5 * age ** 1.5 +  # Depreciation with age
        np.random.randn(n_samples) * 50  # Random noise
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'age_years': age,
        'price_1000s': price
    })
    
    return data

def generate_polynomial_data(n_samples=100, noise=0.5, degree=2, random_state=42):
    """Generate polynomial data with noise."""
    np.random.seed(random_state)
    X = 6 * np.random.rand(n_samples, 1) - 3  # X between -3 and 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(n_samples, 1) * noise
    return X, y

def save_sample_data():
    """Generate and save sample data files."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate and save linear data
    X_linear, y_linear = generate_linear_data(n_samples=200, noise=0.5)
    linear_data = np.hstack((X_linear, y_linear))
    np.savetxt(os.path.join(data_dir, 'linear_data.csv'), 
              linear_data, 
              delimiter=',', 
              header='feature,target',
              comments='',
              fmt='%.4f')
    
    # Generate and save housing data
    housing_data = generate_housing_data(n_samples=500)
    housing_data.to_csv(os.path.join(data_dir, 'housing_data.csv'), index=False)
    
    # Generate and save polynomial data
    X_poly, y_poly = generate_polynomial_data(n_samples=100, noise=1.0, degree=2)
    poly_data = np.hstack((X_poly, y_poly))
    np.savetxt(os.path.join(data_dir, 'polynomial_data.csv'), 
              poly_data, 
              delimiter=',', 
              header='feature,target',
              comments='',
              fmt='%.4f')
    
    print(f"Sample data saved to {data_dir}/")

if __name__ == "__main__":
    save_sample_data()
