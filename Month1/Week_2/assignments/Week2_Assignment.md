# Week 2 Production Engineering Assignment: In-Process Analytical Feature Store & Vector Staging Engine

## Executive Overview & Architectural Motivation
Enterprise AI pipelines often struggle with the "dual-system" problem: machine learning engineers query disparate transactional databases (Postgres/MySQL) via ORMs, dumping flat CSV files, and executing expensive Python joins that exhaust server memory and cause training-serving skew.

In this assignment, you will build an **In-Process Analytical Feature Store & Staging Engine** using **DuckDB**, SQL window semantics, and Apache Arrow integration.

Your system will ingest partitioned parquet transaction records, construct multi-tier temporal aggregation features with zero leakage, join contextual product catalogs hierarchically, stage vectorized embeddings, and expose an ultra-low-latency SQL interface for ML training dataset extraction.

---

## Technical Specifications & Architecture Diagram

```
                 Raw Parquet Event Logs & Product Catalogs
                 [/data/transactions/*.parquet, /data/catalog.parquet]
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          In-Memory DuckDB Engine          │
                  │   - Vectorized Execution Pipeline         │
                  │   - Direct Parquet Memory-Mapped Scanning │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │       Temporal Feature Engineering        │
                  │   - 1h, 24h, 7d Rolling Aggregations      │
                  │   - Sessionization Boundary Detection     │
                  │   - Category Hierarchy Recursive Breadcrumb│
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │         In-Process Vector Staging         │
                  │   - Dot Product Array Operations          │
                  │   - Hybrid SQL Filtering + Vector KNN     │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  Leak-Free ML Training Splits in Apache Arrow Format
```

---

## Detailed Deliverable Requirements

### 1. Zero-Copy Parquet Schema Staging (`staging.py`)
- Initialize a persistent or in-memory DuckDB instance (`duckdb.connect()`).
- Create views directly over partitioned Parquet files using `read_parquet(['path/to/part*.parquet'])` with zero-copy table registration.
- Verify through `EXPLAIN` that DuckDB pushes column projections and filters into the underlying parquet files.

### 2. Time-Travel & Leak-Free Temporal Feature Views (`features.sql`)
Write production SQL scripts implementing:
- **Session Boundary Extraction**: 30-minute inactivity boundary identification via `LAG()` and cumulative window sums.
- **Leak-Free Point-In-Time Window Aggregations**:
  $$\text{Rolling Spend} = \text{SUM}(\text{amount}) \text{ OVER } (\text{PARTITION BY user\_id ORDER BY timestamp RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW})$$
- **Hierarchical Taxonomy Expansion**: Recursive CTE to flatten product category parent-child hierarchies into human-readable path strings.

### 3. In-Process Vector Staging & Cosine Search (`vector_search.py`)
- Store multimodal product embeddings as native DuckDB float arrays (`FLOAT[64]`).
- Execute a hybrid search query combining exact SQL metadata predicates (`category_depth <= 2`, `price BETWEEN 10.0 AND 100.0`) with cosine similarity scoring.

---

## Starter Code Template & Submission Skeleton

```python
"""
duckdb_feature_store.py - Production In-Process Feature Store
"""
import duckdb
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, Any

class AnalyticalFeatureStore:
    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(db_path)
        self.con.execute("SET preserve_insertion_order=false;")

    def register_parquet_sources(self, raw_dir: str):
        """Creates zero-copy views over partitioned parquet event files."""
        pattern = os.path.join(raw_dir, "*.parquet")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_transactions AS SELECT * FROM read_parquet('{pattern}');")

    def build_feature_views(self):
        """Constructs temporal analytical feature views using SQL window functions."""
        feature_sql = """
        CREATE OR REPLACE VIEW user_temporal_features AS
        WITH lagged AS (
            SELECT 
                user_id,
                transaction_id,
                timestamp,
                amount,
                category_id,
                LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_ts
            FROM raw_transactions
        ),
        sessions AS (
            SELECT 
                *,
                CASE WHEN prev_ts IS NULL OR (timestamp - prev_ts) > 1800 THEN 1 ELSE 0 END AS new_sess_flag
            FROM lagged
        ),
        session_indexed AS (
            SELECT 
                *,
                SUM(new_sess_flag) OVER (PARTITION BY user_id ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS session_id
            FROM sessions
        )
        SELECT 
            user_id,
            transaction_id,
            timestamp,
            session_id,
            amount,
            AVG(amount) OVER (
                PARTITION BY user_id 
                ORDER BY timestamp 
                ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
            ) AS rolling_avg_amount_last10,
            COUNT(transaction_id) OVER (
                PARTITION BY user_id, session_id 
                ORDER BY timestamp 
                ROWS UNBOUNDED PRECEDING
            ) AS events_in_current_session
        FROM session_indexed;
        """
        self.con.execute(feature_sql)

    def extract_training_split(self) -> pd.DataFrame:
        """Extracts engineered training dataset into a Pandas/Arrow dataframe."""
        return self.con.execute("SELECT * FROM user_temporal_features ORDER BY timestamp LIMIT 500;").df()

# Verification Run
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic transaction parquet records
        n = 50_000
        df = pd.DataFrame({
            "transaction_id": np.arange(n),
            "user_id": np.random.choice([f"usr_{i}" for i in range(100)], size=n),
            "timestamp": np.sort(np.random.randint(1_700_000_000, 1_700_100_000, size=n)),
            "amount": np.random.uniform(5.0, 500.0, size=n),
            "category_id": np.random.randint(1, 10, size=n)
        })
        df.to_parquet(os.path.join(tmpdir, "transactions_01.parquet"))
        
        store = AnalyticalFeatureStore()
        store.register_parquet_sources(tmpdir)
        store.build_feature_views()
        
        train_df = store.extract_training_split()
        print("Engineered Training Feature Head:")
        print(train_df[["user_id", "session_id", "rolling_avg_amount_last10", "events_in_current_session"]].head(10))
        assert not train_df.empty
        assert "events_in_current_session" in train_df.columns
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **SQL Window Function Correctness** | 30 | Flawless partition and frame declarations (`ROWS BETWEEN ...`), zero forward leakage. |
| **Sessionization & Temporal Integrity** | 25 | Precise threshold detection with cumulative sum monotonic session assignment. |
| **In-Process Vector Operations** | 20 | Fast array dot-product similarity computation with hybrid metadata filtering. |
| **Zero-Copy Parquet Integration** | 15 | Direct memory-mapped query execution over disk parquet without intermediary copying. |
| **Verification & Test Suite** | 10 | Complete Python unit test suite validating schema definitions and query results. |
