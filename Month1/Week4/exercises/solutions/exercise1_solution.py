"""
Exercise 1: SQL for Data Analysis - Solutions
"""

import sqlite3
import pandas as pd

# Exercise 1: Basic Query
def exercise_1():
    """Get all customers who joined in January 2023."""
    query = """
    SELECT *
    FROM customers
    WHERE strftime('%Y-%m', join_date) = '2023-01';
    """
    return query

# Exercise 2: Join Tables
def exercise_2():
    """Get all orders with customer names and email addresses."""
    query = """
    SELECT o.order_id, o.order_date, o.amount, o.status,
           c.name as customer_name, c.email
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id;
    """
    return query

# Exercise 3: Aggregation
def exercise_3():
    """Get total sales by customer, ordered by total spent (highest first)."""
    query = """
    SELECT c.customer_id, c.name, SUM(o.amount) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.name
    ORDER BY total_spent DESC;
    """
    return query

# Exercise 4: Subquery
def exercise_4():
    """Find customers who have spent more than the average customer."""
    query = """
    WITH customer_totals AS (
        SELECT customer_id, SUM(amount) as total_spent
        FROM orders
        GROUP BY customer_id
    )
    SELECT c.customer_id, c.name, ct.total_spent
    FROM customers c
    JOIN customer_totals ct ON c.customer_id = ct.customer_id
    WHERE ct.total_spent > (SELECT AVG(total_spent) FROM customer_totals);
    """
    return query

# Exercise 5: CTE (Common Table Expression)
def exercise_5():
    """Find the most popular product by number of items sold."""
    query = """
    WITH product_sales AS (
        SELECT p.product_id, p.name, SUM(oi.quantity) as total_quantity
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        GROUP BY p.product_id, p.name
    )
    SELECT product_id, name, total_quantity
    FROM product_sales
    WHERE total_quantity = (SELECT MAX(total_quantity) FROM product_sales);
    """
    return query

def setup_database():
    """Set up the sample database for testing."""
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
        (3, 'Bob Johnson', 'bob@example.com', '2023-01-05'),
        (4, 'Alice Brown', 'alice@example.com', '2023-03-01')
    ]
    cursor.executemany('INSERT INTO customers VALUES (?, ?, ?, ?)', customers)
    
    orders = [
        (101, 1, '2023-03-10', 150.0, 'completed'),
        (102, 1, '2023-03-15', 75.5, 'completed'),
        (103, 2, '2023-03-12', 200.0, 'shipped'),
        (104, 3, '2023-03-20', 300.0, 'processing'),
        (105, 4, '2023-03-22', 180.0, 'completed')
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
        (104, 4, 2, 300.0),
        (105, 1, 1, 1000.0),
        (105, 2, 1, 25.0)
    ]
    cursor.executemany('INSERT INTO order_items VALUES (?, ?, ?, ?)', order_items)
    
    conn.commit()
    return conn

def main():
    # Set up the database
    conn = setup_database()
    
    # Test each exercise
    exercises = [
        ("1. Customers who joined in January 2023", exercise_1()),
        ("2. Orders with customer details", exercise_2()),
        ("3. Total sales by customer", exercise_3()),
        ("4. Customers who spent more than average", exercise_4()),
        ("5. Most popular product by quantity sold", exercise_5())
    ]
    
    for title, query in exercises:
        print(f"\n--- {title} ---")
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
