# Month 1, Week 2: Production SQL & Relational Databases for AI Systems

## 1. Executive Summary & Industry Context
While deep learning models and LLMs capture headlines, **the enterprise data layer runs on SQL**. Relational databases (PostgreSQL, Snowflake, BigQuery) store user interactions, financial transactions, and operational logs. 

An AI engineer who cannot write performant SQL is dependent on data analysts to extract training datasets. Furthermore, computing features directly inside the database via **Analytical Window Functions** minimizes data movement, reduces memory consumption, and accelerates feature store preparation.

---

## 2. Canonical Textbook References
* **Designing Data-Intensive Applications (2017)** by Martin Kleppmann, Chapter 3: *Storage and Retrieval — B-Trees and LSM-Trees*.
* **SQL in 10 Minutes, Sams Teach Yourself (5th Edition)** by Ben Forta.

---

## 3. Theoretical Foundations: Relational Models, Indexes & Window Functions

### Relational Schemas & Indexing Mechanics
* **Primary Key:** A unique identifier enforcing entity integrity.
* **Foreign Key:** Enforces referential integrity between parent and child tables.
* **B-Tree Indexes:** When you query `WHERE user_id = 45091`, an unindexed table performs a full sequential scan ($O(N)$ operations). A B-Tree index maintains a balanced multi-level search tree ($O(\log N)$ lookups), jumping straight to the exact disk block.

### The Power of Window Functions
Unlike standard `GROUP BY` aggregations (which collapse multiple rows into a single summary row), **Window Functions** compute aggregate or ranking calculations across a sliding set of table rows while **preserving the original row identity**:

$$\text{FUNCTION}() \text{ OVER } (\text{PARTITION BY } \text{entity} \text{ ORDER BY } \text{timestamp} \text{ [ROWS BETWEEN ...]})$$

* `ROW_NUMBER()`: Unique consecutive integer for each row within a partition.
* `RANK()` / `DENSE_RANK()`: Rankings handling ties with or without gaps.
* `LEAD(col, n)` / `LAG(col, n)`: Accessing data from subsequent or previous rows without expensive self-joins (ideal for time-series feature engineering).

---

## 4. Complete Step-by-Step Code Walkthrough

```sql
-- Production Feature Engineering Script in PostgreSQL / DuckDB
-- Objective: For each user transaction, compute:
-- 1. The user's historical transaction sequence number.
-- 2. The amount of their immediately preceding transaction.
-- 3. The running 7-day cumulative spend.

WITH user_activity_cleaned AS (
    SELECT 
        transaction_id,
        user_id,
        amount,
        created_at,
        category,
        -- Window Function 1: Transaction Sequence Number per User
        ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY created_at ASC
        ) AS user_transaction_seq,
        
        -- Window Function 2: Previous Transaction Amount (Detecting sudden spikes)
        LAG(amount, 1, 0.0) OVER (
            PARTITION BY user_id 
            ORDER BY created_at ASC
        ) AS previous_amount,
        
        -- Window Function 3: Running Cumulative Total Spend
        SUM(amount) OVER (
            PARTITION BY user_id 
            ORDER BY created_at ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_spend_to_date
    FROM transactions
    WHERE status = 'completed'
)
SELECT 
    transaction_id,
    user_id,
    amount,
    previous_amount,
    (amount - previous_amount) AS amount_delta,
    cumulative_spend_to_date,
    user_transaction_seq
FROM user_activity_cleaned
ORDER BY user_id, user_transaction_seq;
```

### Python Integration with DuckDB (In-Process Fast SQL for AI)
```python
import duckdb
import polars as pl

# Connect to in-memory DuckDB
con = duckdb.connect()

# Query Parquet files directly with zero-copy SQL
query = """
    SELECT 
        category,
        COUNT(*) as total_orders,
        AVG(transaction_amount) as mean_spend,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amount) as median_spend
    FROM read_parquet('transactions.parquet')
    GROUP BY category
    ORDER BY mean_spend DESC
"""

result_df: pl.DataFrame = con.execute(query).pl()
print(result_df)
```

---

## 5. Common Student Pitfalls & Debugging Strategies

1. **Cartesian Products from Missing Join Conditions:**
   * *The Mistake:* Writing `FROM users, transactions` without an explicit `ON users.id = transactions.user_id`, causing an explosive cross join ($N \times M$ rows).
   * *The Fix:* Always write explicit `INNER JOIN ... ON ...` or `LEFT JOIN ... ON ...`.
2. **Ignoring Execution Plans (`EXPLAIN ANALYZE`):**
   * *The Mistake:* Assuming a slow query is due to hardware rather than unindexed columns.
   * *The Fix:* Prepend `EXPLAIN ANALYZE` to the query to identify sequential scans (`Seq Scan`) and high-cost joins.
3. **Improper `NULL` Handling:**
   * *The Mistake:* Writing `WHERE status != 'inactive'`, which silently drops rows where `status` is `NULL`.
   * *The Fix:* Use `WHERE status IS DISTINCT FROM 'inactive'` or handle `COALESCE(status, 'unknown')`.

---

## 6. Production Best Practices & Real-World Case Study
Feature engineering in enterprise AI systems (e.g., fraud detection at PayPal or recommendation feeds at Netflix) uses **SQL Feature Pipelines**:
* Real-time transactions are compared against historical user window metrics computed in SQL (e.g., `avg_amount_last_30_days`, `num_failed_logins_last_1_hour`).
* Extracting features inside the database engine minimizes network bandwidth transfer into Python training harnesses.

---

## 7. Hands-on Mini-Lab & Self-Assessment Checklist
- [ ] Install DuckDB: `uv pip install duckdb`
- [ ] Query a local Parquet dataset using SQL with `read_parquet()`.
- [ ] Implement a query using `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` to select the top 3 transactions per user.
- [ ] Use `LAG()` to calculate the time difference between consecutive events for each user.
- [ ] Inspect execution time using `EXPLAIN ANALYZE`.
