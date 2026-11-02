# Month 1, Week 1: High-Performance Data Processing with Polars & Parquet

![Columnar vs Row Storage](../../../docs/images/columnar_vs_row_storage.png)

## 1. Executive Summary & Industry Context
For over a decade, **Pandas** served as the de facto library for tabular data in Python. However, as datasets have scaled from megabytes to gigabytes and terabytes, traditional Pandas workflows face severe architectural bottlenecks:
1. **Single-Threaded Execution:** Pandas operations run on a single CPU core by default.
2. **Eager Evaluation:** Every intermediate operation creates an immediate in-memory copy of the entire DataFrame, easily triggering Out-Of-Memory (OOM) crashes.
3. **Row-Oriented Memory Overhead:** Storing heterogeneous data types creates memory bloat (often 3x to 5x the raw file size on disk).

In modern AI Engineering, high-speed data preparation is powered by **Polars** (written in Rust and built on the Apache Arrow columnar format). Polars provides multi-threaded execution, SIMD vectorization, and lazy query optimization—running operations up to 10x–50x faster than Pandas while using significantly less RAM.

---

## 2. Canonical Textbook References
* **Python for Data Analysis (3rd Edition, 2022)** by Wes McKinney (Creator of Pandas), Chapter 12: *Advanced Pandas & Interoperability*.
* **Designing Data-Intensive Applications (2017)** by Martin Kleppmann, Chapter 3: *Storage and Retrieval — Column-Oriented Storage*.

---

## 3. Theoretical Foundations: Columnar Memory & Lazy Evaluation

### Columnar Memory (Apache Arrow) vs. Row-Oriented Layout
In traditional row-oriented storage (like CSVs or relational OLTP databases), data is stored row-by-row on disk and in memory:
$$\text{Row 0: } [id_0, age_0, salary_0], \quad \text{Row 1: } [id_1, age_1, salary_1], \quad \dots$$

If you want to compute the average `salary` across 10 million rows, a row-oriented system must load the *entire* row into CPU cache, discarding $id$ and $age$.

In **Columnar Storage (Apache Arrow & Parquet)**, contiguous memory blocks store individual columns:
$$\text{Column } salary: [salary_0, salary_1, \dots, salary_N]$$

* **Cache Locality:** The CPU cache is populated exclusively with contiguous salary values.
* **SIMD (Single Instruction, Multiple Data):** Modern CPU vector registers process 4, 8, or 16 numbers in a single clock cycle.
* **Compression Efficiency:** Columnar data of uniform types compresses dramatically better (using run-length encoding, dictionary encoding, and Snappy compression).

### Eager vs. Lazy Evaluation
* **Eager Evaluation (Pandas):** Step 1 runs and writes to RAM $\to$ Step 2 runs and writes to RAM $\to$ Step 3 runs.
* **Lazy Evaluation (Polars `LazyFrame`):** You define a chain of expressions. Polars creates a **Logical Query Plan**, optimizes it (pushing down predicates to filter rows *before* loading columns), and executes the graph in parallel across all CPU cores.

---

## 4. Complete Step-by-Step Code Walkthrough

```python
import polars as pl
import time
import numpy as np

# 1. Generating a 5-million row synthetic dataset
n_rows = 5_000_000
np.random.seed(42)

df_raw = pl.DataFrame({
    "user_id": np.arange(n_rows),
    "category": np.random.choice(["Electronics", "Fashion", "Home", "Sports", "Books"], n_rows),
    "transaction_amount": np.random.uniform(5.0, 500.0, n_rows),
    "is_fraud": np.random.choice([0, 1], n_rows, p=[0.98, 0.02]),
    "country_code": np.random.choice(["US", "UK", "CA", "DE", "NG", "IN"], n_rows)
})

# 2. Save as columnar Parquet file with Snappy compression
parquet_file = "transactions.parquet"
df_raw.write_parquet(parquet_file, compression="snappy")
print(f"Saved {n_rows:,} rows to '{parquet_file}'.")

# 3. Fast Lazy Query Execution Plan
# We want: For non-fraudulent US transactions, compute the mean and max amount by category.
start_time = time.time()

query_plan = (
    pl.scan_parquet(parquet_file)  # Lazy scan: does not read data into RAM yet!
    .filter((pl.col("country_code") == "US") & (pl.col("is_fraud") == 0))  # Predicate Pushdown
    .group_by("category")
    .agg([
        pl.col("transaction_amount").count().alias("transaction_count"),
        pl.col("transaction_amount").mean().alias("avg_amount"),
        pl.col("transaction_amount").max().alias("max_amount"),
        pl.col("transaction_amount").std().alias("std_amount")
    ])
    .sort("avg_amount", descending=True)
)

# Inspect the optimized physical execution graph
print("\n--- Optimized Logical Plan ---")
print(query_plan.explain())

# Execute the query in parallel across CPU cores
result = query_plan.collect()
elapsed = time.time() - start_time

print(f"\nQuery completed in {elapsed:.4f} seconds!")
print(result)
```

---

## 5. Common Student Pitfalls & Debugging Strategies

1. **Calling `.collect()` Too Early:**
   * *The Mistake:* Calling `.collect()` immediately after reading data, converting the LazyFrame into an eager in-memory DataFrame.
   * *The Fix:* Keep the operations as a `LazyFrame` (`pl.scan_parquet()`) for all filtering, grouping, and transformations; call `.collect()` only at the very end when you need the final result.
2. **Using Python Loops (`for row in df`):**
   * *The Mistake:* Iterating through rows in Python loops. This serializes data out of Rust/Arrow into slow Python objects, destroying performance.
   * *The Fix:* Always write **Polars Expressions** (`pl.col("x") * 2`), which execute at compiled C++/Rust speeds.
3. **Memory Crashes with Multi-Gigabyte Files:**
   * *The Fix:* Use Polars Streaming mode (`query_plan.collect(streaming=True)`), which processes data in chunks without exceeding RAM.

---

## 6. Production Best Practices & Real-World Case Study
At enterprise scale (e.g., Stripe, Uber, DoorDash), raw event streams generate petabytes of data. Feature stores do not store training data in CSV format:
* **Parquet with Column Pruning:** If a dataset has 200 features, an engineer training a model on 15 features queries *only* those 15 columns directly from storage without downloading the other 185 columns.
* **Partitioned Datasets:** Data is partitioned by date (`year=2026/month=03/day=23/`), allowing queries to scan only relevant temporal slices.

---

## 7. Hands-on Mini-Lab & Self-Assessment Checklist
- [ ] Install Polars: `uv pip install polars pyarrow`
- [ ] Create a synthetic dataset with at least 1,000,000 rows.
- [ ] Write the dataset to both `dataset.csv` and `dataset.parquet`. Compare file sizes on disk.
- [ ] Write a Polars lazy expression using `pl.scan_parquet()` that filters by 2 criteria and computes group statistics.
- [ ] Print the execution plan using `.explain()`. Verify that predicate pushdown is active.
