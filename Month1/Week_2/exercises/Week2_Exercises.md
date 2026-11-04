# Week 2: Graduate-Level Problem Set — Production SQL, DuckDB & Analytical Window Engineering

Welcome to the Month 1 Week 2 Problem Set. This suite covers the modern database engineering paradigms used in enterprise AI architectures: analytical window functions, recursive common table expressions (CTEs), columnar vectorized SQL execution engines (DuckDB), in-process vector staging, and query plan optimization.

---

## Tier 1: Analytical Window Functions & Cumulative Analytics

### Exercise 1: Sessionization via Window Functions & Cumulative Sum Flags
**Concepts**: `LEAD`, `LAG`, `PARTITION BY`, dynamic sessionization boundaries.

#### Problem Specification
In user activity modeling, clickstreams must be segmented into discrete engagement sessions where any inactivity gap exceeding 30 minutes triggers a new session:
1. Given a sequence of raw user click logs `(user_id, timestamp, page_url)`.
2. Compute the temporal delta $\Delta t = t_i - t_{i-1}$ using `LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp)`.
3. Generate a binary session boundary flag: $b_i = 1$ if $\Delta t > 1800 \text{ seconds}$ else $0$.
4. Compute the monotonic `session_id` using a cumulative window sum: $\sum b_i$.
5. Write the complete SQL query and execute it in an in-memory DuckDB database on 10,000 synthetic click events.

#### Verification Solution
```python
import duckdb
import pandas as pd
import numpy as np

def verify_sessionization_sql():
    con = duckdb.connect(database=":memory:")
    
    # Generate 100 synthetic click events for a single user
    np.random.seed(42)
    timestamps = [1_700_000_000]
    for _ in range(99):
        # 85% short clicks (10s - 300s), 15% long pauses (2000s - 5000s)
        gap = int(np.random.choice([np.random.randint(10, 300), np.random.randint(2000, 5000)], p=[0.85, 0.15]))
        timestamps.append(timestamps[-1] + gap)
        
    df = pd.DataFrame({
        "user_id": ["user_123"] * 100,
        "timestamp": timestamps,
        "page_url": [f"/page_{np.random.randint(1, 10)}" for _ in range(100)]
    })
    con.register("raw_clicks", df)
    
    query = """
    WITH lagged_clicks AS (
        SELECT 
            user_id,
            timestamp,
            page_url,
            LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_timestamp
        FROM raw_clicks
    ),
    session_flags AS (
        SELECT 
            user_id,
            timestamp,
            page_url,
            CASE 
                WHEN prev_timestamp IS NULL THEN 1
                WHEN (timestamp - prev_timestamp) > 1800 THEN 1
                ELSE 0 
            END AS is_new_session
        FROM lagged_clicks
    )
    SELECT 
        user_id,
        timestamp,
        page_url,
        SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS session_id
    FROM session_flags
    ORDER BY timestamp;
    """
    
    result = con.execute(query).df()
    print("Sessionization Output Head:")
    print(result.head(10))
    total_sessions = result["session_id"].nunique()
    print(f"Total discrete user sessions detected: {total_sessions}")
    assert total_sessions > 1
    assert result["session_id"].iloc[0] == 1

if __name__ == "__main__":
    verify_sessionization_sql()
```

---

### Exercise 2: Cumulative Moving Aggregations & Quantile Bucketing
**Concepts**: Window frames (`ROWS BETWEEN N PRECEDING AND CURRENT ROW`), `NTILE(k)`, rank percentiles.

#### Problem Specification
1. In feature engineering for financial fraud models, compute:
   - 7-day rolling transaction volume per credit card.
   - 30-day cumulative expenditure.
   - Decile ranking (`NTILE(10)`) of transaction amounts relative to all transactions in that merchant category.
2. Formulate the query in DuckDB SQL using window frames.
3. Validate that no future data leaks into the window calculation (`RANGE BETWEEN ... PRECEDING`).

#### Verification Solution
```python
import duckdb
import pandas as pd
import numpy as np

def verify_window_frames():
    con = duckdb.connect(database=":memory:")
    
    # Generate transactions
    n = 1000
    df = pd.DataFrame({
        "card_id": np.random.choice(["card_A", "card_B", "card_C"], size=n),
        "amount": np.random.exponential(scale=50.0, size=n),
        "category": np.random.choice(["RETAIL", "GROCERY", "TRAVEL"], size=n),
        "txn_day": np.sort(np.random.randint(1, 31, size=n))
    })
    con.register("transactions", df)
    
    query = """
    SELECT 
        card_id,
        category,
        txn_day,
        amount,
        AVG(amount) OVER (
            PARTITION BY card_id 
            ORDER BY txn_day 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7day_avg,
        SUM(amount) OVER (
            PARTITION BY card_id 
            ORDER BY txn_day 
            ROWS UNBOUNDED PRECEDING
        ) AS cumulative_amount,
        NTILE(10) OVER (
            PARTITION BY category 
            ORDER BY amount
        ) AS category_decile
    FROM transactions
    ORDER BY card_id, txn_day
    LIMIT 20;
    """
    res = con.execute(query).df()
    print(res.head(10))
    assert not res.isna().any().any()

if __name__ == "__main__":
    verify_window_frames()
```

---

## Tier 2: Recursive CTEs & Graph Traversal

### Exercise 3: Hierarchical Category Tree Traversal via Recursive CTE
**Concepts**: Recursive Common Table Expressions, graph reachability, depth termination.

#### Problem Specification
In e-commerce item taxonomies, product categories form a directed tree (e.g., Electronics -> Audio -> Headphones -> Wireless).
1. Given a relation `categories(category_id, parent_id, name)`.
2. Construct a recursive CTE in DuckDB SQL that:
   - Traverses from root categories down to all leaf categories.
   - Computes the full breadcrumb path (e.g., `"Electronics > Audio > Headphones"`).
   - Computes the taxonomy tree depth level.
3. Assert that the recursive loop terminates gracefully and detects cyclical references.

#### Verification Solution
```python
import duckdb
import pandas as pd

def verify_recursive_cte():
    con = duckdb.connect(database=":memory:")
    
    tax_data = pd.DataFrame([
        {"id": 1, "parent_id": None, "name": "Electronics"},
        {"id": 2, "parent_id": 1, "name": "Computers"},
        {"id": 3, "parent_id": 1, "name": "Audio"},
        {"id": 4, "parent_id": 2, "name": "Laptops"},
        {"id": 5, "parent_id": 2, "name": "Workstations"},
        {"id": 6, "parent_id": 3, "name": "Headphones"},
        {"id": 7, "parent_id": 6, "name": "Noise Cancelling"}
    ])
    con.register("categories", tax_data)
    
    query = """
    WITH RECURSIVE category_tree AS (
        -- Anchor Member: Root categories
        SELECT 
            id, 
            parent_id, 
            name, 
            CAST(name AS VARCHAR) AS breadcrumb,
            1 AS level
        FROM categories
        WHERE parent_id IS NULL
        
        UNION ALL
        
        -- Recursive Member
        SELECT 
            c.id, 
            c.parent_id, 
            c.name,
            CAST(ct.breadcrumb || ' > ' || c.name AS VARCHAR) AS breadcrumb,
            ct.level + 1 AS level
        FROM categories c
        JOIN category_tree ct ON c.parent_id = ct.id
    )
    SELECT * FROM category_tree ORDER BY level, id;
    """
    res = con.execute(query).df()
    print("Taxonomy Traversal Output:")
    print(res[["id", "breadcrumb", "level"]])
    assert res[res["id"] == 7]["breadcrumb"].iloc[0] == "Electronics > Audio > Headphones > Noise Cancelling"
    assert res[res["id"] == 7]["level"].iloc[0] == 4

if __name__ == "__main__":
    verify_recursive_cte()
```

---

## Tier 3: In-Process Vector Staging & Cosine Distance in DuckDB

### Exercise 4: High-Dimensional Vector Similarity Search in DuckDB
**Concepts**: Array types in DuckDB, cosine similarity SQL expression, vector indexing.

#### Problem Specification
Modern AI applications frequently require fast in-process hybrid search combining relational SQL filters with vector similarity rankings without standing up external databases:
1. In DuckDB, create a table storing item embeddings as `FLOAT[128]` (128-dimensional float arrays).
2. Write a SQL function/expression that computes exact Cosine Similarity:
   $$\text{cosine\_sim}(u, v) = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$
3. Execute a hybrid query: filter items by metadata (`category == 'Laptops' AND price < 1500.00`) and rank the Top-5 nearest neighbors to a query vector.

#### Verification Solution
```python
import duckdb
import numpy as np

def verify_duckdb_vector_search():
    con = duckdb.connect(database=":memory:")
    
    # Generate 1,000 items with 16-dim embeddings for testing
    np.random.seed(42)
    dim = 16
    n = 1000
    
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    con.execute("CREATE TABLE products (id INTEGER, category VARCHAR, price DOUBLE, embedding FLOAT[16])")
    
    query_vec = np.random.randn(dim).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    
    for i in range(n):
        cat = "Laptops" if i % 3 == 0 else "Phones"
        price = 500.0 + (i * 2.5) % 1500.0
        vec = embeddings[i].tolist()
        con.execute("INSERT INTO products VALUES (?, ?, ?, ?)", [i, cat, price, vec])
        
    con.register("target_vec", np.array([query_vec.tolist()]))
    
    # DuckDB array cosine similarity via list_inner_product
    query = """
    SELECT 
        id, 
        category, 
        price,
        list_inner_product(embedding, [?]::FLOAT[16]) AS dot_similarity
    FROM products
    WHERE category = 'Laptops' AND price < 1200.0
    ORDER BY dot_similarity DESC
    LIMIT 5;
    """
    res = con.execute(query, [query_vec.tolist()]).df()
    print("Top 5 Nearest Neighbor Hybrid Results:")
    print(res)
    assert len(res) == 5
    assert (res["price"] < 1200.0).all()

if __name__ == "__main__":
    verify_duckdb_vector_search()
```
