"""
Create a sample SQLite database with housing data for linear regression exercises.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_housing_data(n_samples=1000):
    """Generate synthetic housing data."""
    # Generate features
    data = {
        'id': range(1, n_samples + 1),
        'price': np.random.normal(350000, 150000, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 7, n_samples),
        'bathrooms': np.round(np.random.uniform(1, 4, n_samples), 1),
        'sqft_living': np.random.normal(2000, 800, n_samples).astype(int),
        'sqft_lot': np.random.normal(15000, 10000, n_samples).astype(int),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.normal(7, 1, n_samples).round().clip(1, 13).astype(int),
        'sqft_above': 0,
        'sqft_basement': 0,
        'yr_built': np.random.randint(1900, 2023, n_samples),
        'yr_renovated': 0,
        'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005, 98006], n_samples),
        'lat': np.random.normal(47.5, 0.2, n_samples).round(6),
        'long': np.random.normal(-122.2, 0.2, n_samples).round(6),
        'sqft_living15': 0,
        'sqft_lot15': 0,
        'date': [(datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y%m%dT000000') 
                for _ in range(n_samples)]
    }
    
    # Calculate derived features
    data['sqft_above'] = (data['sqft_living'] * np.random.uniform(0.7, 1.0, n_samples)).astype(int)
    data['sqft_basement'] = data['sqft_living'] - data['sqft_above']
    data['sqft_living15'] = (data['sqft_living'] * np.random.uniform(0.8, 1.2, n_samples)).astype(int)
    data['sqft_lot15'] = (data['sqft_lot'] * np.random.uniform(0.8, 1.2, n_samples)).astype(int)
    
    # Some houses have been renovated
    renovated = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['yr_renovated'] = np.where(
        renovated == 1,
        (data['yr_built'] + np.random.randint(1, 50, n_samples)).clip(max=2022),
        0
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure no negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    return df

def create_database(df, db_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'housing.db'))):
    """Create SQLite database and load data."""
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Write the data to a SQLite table
    df.to_sql('houses', conn, if_exists='replace', index=False)
    
    # Create additional tables for normalization
    # 1. Zipcode lookup table
    zipcodes = df[['zipcode', 'lat', 'long']].drop_duplicates()
    zipcodes.to_sql('zipcodes', conn, if_exists='replace', index=False)
    
    # 2. House features
    house_features = df[['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement']]
    house_features.to_sql('house_features', conn, if_exists='replace', index=False)
    
    # 3. House transactions
    transactions = df[['id', 'price', 'date']]
    transactions.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Create indexes for better query performance
    with conn:
        conn.execute('CREATE INDEX idx_zipcode ON houses(zipcode)')
        conn.execute('CREATE INDEX idx_price ON houses(price)')
        conn.execute('CREATE INDEX idx_bedrooms ON houses(bedrooms)')
        conn.execute('CREATE INDEX idx_bathrooms ON houses(bathrooms)')
    
    # Close the connection
    conn.close()
    
    print(f"Database created successfully at {db_path}")
    print(f"Total records: {len(df)}")

def main():
    # Generate sample data
    print("Generating housing data...")
    housing_data = generate_housing_data(n_samples=2000)
    
    # Create database
    print("Creating SQLite database...")
    create_database(housing_data)
    
    # Save as CSV for reference
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'housing_data_extended.csv'))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    housing_data.to_csv(csv_path, index=False)
    print(f"Sample data saved to {csv_path}")
    
    # Print sample queries
    print("\nSample SQL Queries:")
    print("""
-- Get average price by number of bedrooms
SELECT bedrooms, AVG(price) as avg_price, COUNT(*) as count
FROM houses
GROUP BY bedrooms
ORDER BY bedrooms;

-- Get price distribution by zipcode
SELECT zipcode, 
       COUNT(*) as num_houses,
       MIN(price) as min_price,
       AVG(price) as avg_price,
       MAX(price) as max_price
FROM houses
GROUP BY zipcode
ORDER BY avg_price DESC;

-- Find houses with best value (low price per sqft)
SELECT id, price, sqft_living, 
       ROUND(price*1.0/sqft_living, 2) as price_per_sqft,
       bedrooms, bathrooms
FROM houses
WHERE price > 0 AND sqft_living > 0
ORDER BY price_per_sqft ASC
LIMIT 10;
    """)

if __name__ == "__main__":
    main()
