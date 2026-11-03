# Month 1, Week 2 Cheatsheet: Production SQL, Window Functions & DuckDB Vector Staging

---

## 1. Advanced Analytical Window Functions

Window functions perform calculations across related row partitions without collapsing individual record identities.

### Standard Anatomy
```sql
FUNCTION(...) OVER (
    PARTITION BY partition_col
    ORDER BY sort_col ASC/DESC
    ROWS/RANGE BETWEEN frame_start AND frame_end
)
```

### Essential Navigation & Ranking Functions
- `ROW_NUMBER()`: Unique sequential integer per row in partition ($1, 2, 3, 4$).
- `RANK()`: Assigns identical rank to ties, skipping subsequent ranks ($1, 2, 2, 4$).
- `DENSE_RANK()`: Assigns identical rank to ties without skipping ($1, 2, 2, 3$).
- `NTILE(n)`: Divides partition into $n$ balanced quantile buckets ($1 \dots n$).
- `LAG(col, offset, default)`: Accesses row $k$ positions prior.
- `LEAD(col, offset, default)`: Accesses row $k$ positions forward.

---

## 2. Dynamic Sessionization via Window Cumulative Sums

Identifies user session boundaries whenever elapsed inactivity exceeds $T = 1800\text{ seconds}$:
```sql
WITH lagged_clicks AS (
    SELECT 
        user_id,
        timestamp,
        url,
        LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_ts
    FROM user_events
),
session_flags AS (
    SELECT 
        *,
        CASE 
            WHEN prev_ts IS NULL OR (timestamp - prev_ts) > 1800 THEN 1 
            ELSE 0 
        END AS is_new_session
    FROM lagged_clicks
)
SELECT 
    user_id,
    timestamp,
    url,
    -- Monotonically increasing session ID per user
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS session_id
FROM session_flags;
```

---

## 3. Recursive Common Table Expressions (CTEs)

Traverses hierarchical Directed Acyclic Graphs (DAGs) and category trees:
```sql
WITH RECURSIVE category_tree AS (
    -- 1. Anchor Member: Root nodes with no parent
    SELECT 
        category_id,
        parent_id,
        name,
        CAST(name AS VARCHAR) AS breadcrumb_path,
        1 AS tree_depth
    FROM product_categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- 2. Recursive Member: Joins children with parent path
    SELECT 
        c.category_id,
        c.parent_id,
        c.name,
        CAST(ct.breadcrumb_path || ' > ' || c.name AS VARCHAR) AS breadcrumb_path,
        ct.tree_depth + 1 AS tree_depth
    FROM product_categories c
    JOIN category_tree ct ON c.parent_id = ct.category_id
)
SELECT * FROM category_tree ORDER BY tree_depth, category_id;
```

---

## 4. In-Process Analytical Engine: DuckDB Architecture

DuckDB is a columnar, vectorized SQL database embedded directly in Python:
- **Vectorized Execution**: Processes batches of 2,048 values per vector instruction in CPU L1/L2 cache.
- **Zero-Copy Arrow Integration**: Directly queries Arrow tables, Polars DataFrames, and Pandas DataFrames.

### Direct Parquet Scanning & Query Plan Inspection
```python
import duckdb

con = duckdb.connect(database=":memory:")

# Direct scan of multi-GB Parquet datasets without loading into RAM
con.execute("""
    EXPLAIN ANALYZE
    SELECT 
        category_id,
        AVG(price) AS mean_price,
        COUNT(*) AS total_items
    FROM read_parquet('/data/catalog/*.parquet')
    WHERE in_stock = true
    GROUP BY category_id
    HAVING total_items > 100;
""")
print(con.fetchall())
```

---

## 5. In-Process Vector Staging & Cosine Similarity in DuckDB

DuckDB natively supports fixed-size array types (`FLOAT[d]`) for staging embeddings:
```sql
CREATE TABLE product_embeddings (
    item_id INTEGER PRIMARY KEY,
    category VARCHAR,
    price DOUBLE,
    embedding FLOAT[128]
);

-- Hybrid search: SQL metadata filter + Vector Dot Product
SELECT 
    item_id, 
    category, 
    price,
    list_inner_product(embedding, ?::FLOAT[128]) AS similarity_score
FROM product_embeddings
WHERE category = 'Enterprise Hardware' AND price < 5000.00
ORDER BY similarity_score DESC
LIMIT 5;
```
