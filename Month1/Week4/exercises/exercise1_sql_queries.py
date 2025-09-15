"""
Exercise 1: SQL for Data Analysis

This exercise covers:
- Basic SQL queries
- Joins and aggregations
- Subqueries and CTEs
- Working with SQL in Python
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Sample database setup
def setup_database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        join_date DATE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date DATE,
        amount REAL,
        status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL,
        category TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE order_items (
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        price REAL,
        PRIMARY KEY (order_id, product_id),
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')
    
    # Insert sample data
    customers = [
        (1, 'John Doe', 'john@example.com', '2023-01-15'),
        (2, 'Jane Smith', 'jane@example.com', '2023-02-20'),
        (3, 'Bob Johnson', 'bob@example.com', '2023-01-05')
    ]
    cursor.executemany('INSERT INTO customers VALUES (?, ?, ?, ?)', customers)
    
    orders = [
        (101, 1, '2023-03-10', 150.0, 'completed'),
        (102, 1, '2023-03-15', 75.5, 'completed'),
        (103, 2, '2023-03-12', 200.0, 'shipped'),
        (104, 3, '2023-03-20', 300.0, 'processing')
    ]
    cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', orders)
    
    products = [
        (1, 'Laptop', 1000.0, 'Electronics'),
        (2, 'Mouse', 25.0, 'Electronics'),
        (3, 'Desk', 200.0, 'Furniture'),
        (4, 'Chair', 150.0, 'Furniture')
    ]
    cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', products)
    
    order_items = [
        (101, 1, 1, 1000.0),
        (101, 2, 2, 25.0),
        (102, 3, 1, 200.0),
        (103, 1, 2, 2000.0),
        (104, 4, 2, 300.0)
    ]
    cursor.executemany('INSERT INTO order_items VALUES (?, ?, ?, ?)', order_items)
    
    conn.commit()
    return conn

# Exercise 1: Basic Query
# Write a SQL query to get all customers who joined in January 2023
def exercise_1():
    query = """
    -- Your SQL query here
    """
    return query

# Exercise 2: Join Tables
# Write a SQL query to get all orders with customer names and email addresses
def exercise_2():
    query = """
    -- Your SQL query here
    """
    return query

# Exercise 3: Aggregation
# Write a SQL query to get total sales by customer, ordered by total spent (highest first)
def exercise_3():
    query = """
    -- Your SQL query here
    """
    return query

# Exercise 4: Subquery
# Write a SQL query to find customers who have spent more than the average customer
def exercise_4():
    query = """
    -- Your SQL query here
    """
    return query

# Exercise 5: CTE (Common Table Expression)
# Write a SQL query using a CTE to find the most popular product by number of items sold
def exercise_5():
    query = """
    -- Your SQL query here
    """
    return query

# Main function to test the queries
def main():
    # Set up the database
    conn = setup_database()
    
    # Test each exercise
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n--- Exercise {i} ---")
        query = exercise()
        print(f"Query:\n{query}")
        
        try:
            result = pd.read_sql_query(query, conn)
            print("\nResult:")
            print(result)
        except Exception as e:
            print(f"\nError executing query: {e}")
    
    conn.close()

if __name__ == "__main__":
    main()
