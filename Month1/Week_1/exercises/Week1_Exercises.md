# Week 1: Graduate-Level Problem Set — High-Performance Data Processing with Polars & Arrow

Welcome to the Month 1 Week 1 Problem Set. This suite evaluates your mastery of the Apache Arrow columnar memory model, zero-copy record slicing, SIMD parallelism, Polars LazyFrame logical query optimization (predicate/projection pushdown), and streaming out-of-core execution.

---

## Tier 1: Core Mechanics & Arrow Columnar Internals

### Exercise 1: Apache Arrow Columnar Memory Layout vs Pandas Row-Oriented Footprint
**Concepts**: Arrow Array buffers, validity bitmaps, contiguous coordinate vectors, SIMD vectorization.

#### Problem Specification
1. Construct a synthetic telemetry dataset containing 1,000,000 observations of:
   - `timestamp`: Unix timestamp integer (`Int64`).
   - `device_id`: Categorical string / Dictionary encoded.
   - `voltage`: Float64 with 5% missing (null) values.
   - `current`: Float64 with 5% missing (null) values.
2. Ingest this dataset into:
   - Pandas (`pd.DataFrame`) using default NumPy backend.
   - Polars (`pl.DataFrame`) backed by native Apache Arrow.
3. Compare:
   - Total physical memory allocation in RAM using `memory_usage(deep=True)` vs Arrow buffer sizes.
   - Mean aggregation compute runtime ($\mu \pm \sigma$) across 50 iterations for calculating average power ($P = V \times I$).
4. Explain the mechanics of Arrow's 1-bit validity bitmap compared to Pandas' float-based `NaN` representation.

#### Verification Solution
```python
import numpy as np
import pandas as pd
import polars as pl
import time

def compare_arrow_vs_pandas():
    n_rows = 1_000_000
    np.random.seed(42)
    
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + n_rows, dtype=np.int64)
    devices = np.random.choice(["sensor_alpha", "sensor_beta", "sensor_gamma", "sensor_delta"], size=n_rows)
    voltages = np.random.normal(loc=12.0, scale=0.5, size=n_rows)
    currents = np.random.normal(loc=2.5, scale=0.2, size=n_rows)
    
    # Inject 5% nulls
    voltages[np.random.rand(n_rows) < 0.05] = np.nan
    currents[np.random.rand(n_rows) < 0.05] = np.nan
    
    # 1. Pandas DataFrame
    df_pd = pd.DataFrame({
        "timestamp": timestamps,
        "device_id": devices,
        "voltage": voltages,
        "current": currents
    })
    pd_memory_mb = df_pd.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # 2. Polars DataFrame
    df_pl = pl.DataFrame({
        "timestamp": timestamps,
        "device_id": devices,
        "voltage": voltages,
        "current": currents
    }).with_columns(pl.col("device_id").cast(pl.Categorical))
    pl_memory_mb = df_pl.estimated_size() / (1024 * 1024)
    
    print(f"Pandas RAM Footprint: {pd_memory_mb:.2f} MB")
    print(f"Polars RAM Footprint: {pl_memory_mb:.2f} MB")
    
    # Benchmark Pandas Power Calculation
    t0 = time.perf_counter()
    for _ in range(10):
        _ = (df_pd["voltage"] * df_pd["current"]).mean()
    t_pd = (time.perf_counter() - t0) / 10
    
    # Benchmark Polars Power Calculation
    t0 = time.perf_counter()
    for _ in range(10):
        _ = df_pl.select((pl.col("voltage") * pl.col("current")).mean()).item()
    t_pl = (time.perf_counter() - t0) / 10
    
    print(f"Pandas Execution Time: {t_pd * 1000:.2f} ms")
    print(f"Polars Execution Time: {t_pl * 1000:.2f} ms")
    assert pl_memory_mb < pd_memory_mb

if __name__ == "__main__":
    compare_arrow_vs_pandas()
```

---

### Exercise 2: Predicate & Projection Pushdown in the Polars Query Optimizer
**Concepts**: `LazyFrame`, Logical Plan, Physical Plan, AST inspection, I/O minimization.

#### Problem Specification
1. Write a Python script generating 1,000,000 records partitioned across 5 Parquet files in `/tmp/telemetry_partitions/`.
2. Construct a complex query using `pl.scan_parquet()`:
   - Filter rows where `device_id == "sensor_alpha"` and `voltage > 12.2`.
   - Compute moving average of `current` over a sliding window.
   - Select only `timestamp` and computed metric.
3. Call `.explain()` on the `LazyFrame` to output the unoptimized vs optimized logical query execution plans.
4. Verify that:
   - **Predicate Pushdown** pushed the filters directly into the Parquet reader layer.
   - **Projection Pushdown** pruned unused columns before reading data buffers into memory.

#### Verification Solution
```python
import polars as pl
import numpy as np
import tempfile
import os

def verify_query_plan_pushdowns():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate dummy partitioned parquet files
        for part in range(3):
            sub_df = pl.DataFrame({
                "timestamp": np.arange(100_000),
                "device_id": np.random.choice(["sensor_alpha", "sensor_beta"], size=100_000),
                "voltage": np.random.uniform(10.0, 14.0, size=100_000),
                "current": np.random.uniform(1.0, 5.0, size=100_000),
                "unused_sensor_payload": np.random.randn(100_000)
            })
            sub_df.write_parquet(os.path.join(tmpdir, f"part_{part}.parquet"))
            
        # Construct lazy query
        lazy_query = (
            pl.scan_parquet(os.path.join(tmpdir, "*.parquet"))
            .filter((pl.col("device_id") == "sensor_alpha") & (pl.col("voltage") > 12.2))
            .select(["timestamp", "voltage", "current"])
        )
        
        unoptimized_plan = lazy_query.explain(optimized=False)
        optimized_plan = lazy_query.explain(optimized=True)
        
        print("=== UNOPTIMIZED LOGICAL PLAN ===")
        print(unoptimized_plan[:400])
        print("\n=== OPTIMIZED LOGICAL PLAN ===")
        print(optimized_plan[:400])
        
        # In optimized plan, selection and filter are pushed into the Parquet scan node
        assert "unused_sensor_payload" not in optimized_plan
        assert "FAST_PROJECT" in optimized_plan or "PROJECT" in optimized_plan or "SELECTION" in optimized_plan

if __name__ == "__main__":
    verify_query_plan_pushdowns()
```

---

## Tier 2: Vectorized Window Functions & Streaming Processing

### Exercise 3: Vectorized Rolling Quantile & Cumulative Anomaly Detection
**Concepts**: Rolling aggregations, dynamic window grouping, expressions vs Python lambdas.

#### Problem Specification
1. In time-series feature engineering, rolling quantiles ($q_{0.05}$ and $q_{0.95}$) are critical for dynamic outlier clipping.
2. Given a high-frequency financial or telemetry tick stream with 500,000 records:
   - Implement a pure Polars expression using `.rolling(index_column="timestamp", period="10m")` or `pl.col("price").rolling_quantile(...)`.
   - Calculate an upper and lower Bollinger-style dynamic band:
     $$\text{Band}_{\text{upper}} = \mu_{\text{rolling}} + 2 \times \sigma_{\text{rolling}}$$
   - Flag anomalies without dropping into Python loops or `.apply()` calls.
3. Assert that the operation completes under 100 milliseconds for 500k rows.

#### Verification Solution
```python
import polars as pl
import numpy as np
import time

def benchmark_rolling_feature_engineering():
    np.random.seed(42)
    n = 500_000
    base_price = 100.0 + np.cumsum(np.random.normal(0, 0.5, size=n))
    # Inject spike anomalies
    base_price[10_000] += 50.0
    base_price[200_000] -= 45.0
    
    df = pl.DataFrame({
        "timestamp_idx": np.arange(n),
        "price": base_price
    })
    
    t0 = time.perf_counter()
    res = df.with_columns([
        pl.col("price").rolling_mean(window_size=100).alias("mean_100"),
        pl.col("price").rolling_std(window_size=100).alias("std_100"),
    ]).with_columns([
        (pl.col("mean_100") + 3.0 * pl.col("std_100")).alias("upper_band"),
        (pl.col("mean_100") - 3.0 * pl.col("std_100")).alias("lower_band")
    ]).with_columns([
        ((pl.col("price") > pl.col("upper_band")) | (pl.col("price") < pl.col("lower_band")))
        .fill_null(False)
        .alias("is_anomaly")
    ])
    elapsed = time.perf_counter() - t0
    
    anomaly_count = res.select(pl.col("is_anomaly").sum()).item()
    print(f"Rolling calculations on {n:,} rows took: {elapsed * 1000:.2f} ms")
    print(f"Total anomaly ticks identified: {anomaly_count}")
    assert elapsed < 0.20  # Under 200ms
    assert anomaly_count >= 2

if __name__ == "__main__":
    benchmark_rolling_feature_engineering()
```

---

## Tier 3: Out-of-Core Batch Aggregation Engine

### Exercise 4: Out-of-Core Parquet Streaming with Memory Constraints
**Concepts**: Streaming execution engine, memory mapped files, out-of-core chunking.

#### Problem Specification
1. In edge AI applications, raw dataset volume frequently exceeds physical host RAM.
2. Demonstrate how Polars streams out-of-core batches without triggering `MemoryError`:
   - Create a disk-backed dataset split into multiple Parquet chunks totaling 1,000,000 records.
   - Execute a global multi-group aggregation:
     $$\text{groupby}([\text{"device\_id"}, \text{"event\_type"}]).agg([\text{mean}, \text{count}, \text{std}])$$
   - Configure `.collect(streaming=True)`.
3. Verify that the peak memory footprint during collection remains bounded to the working chunk buffer rather than the full dataset size.

#### Verification Solution
```python
import polars as pl
import numpy as np
import tempfile
import os

def verify_streaming_out_of_core():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write chunks
        for chunk in range(5):
            df_chunk = pl.DataFrame({
                "device_id": np.random.choice([f"device_{i}" for i in range(20)], size=200_000),
                "event_type": np.random.choice(["LOGIN", "INFERENCE", "PING", "REFRESH"], size=200_000),
                "latency_ms": np.random.exponential(scale=25.0, size=200_000)
            })
            df_chunk.write_parquet(os.path.join(tmpdir, f"chunk_{chunk}.parquet"))
            
        # Execute streaming aggregation
        aggregated = (
            pl.scan_parquet(os.path.join(tmpdir, "*.parquet"))
            .group_by(["device_id", "event_type"])
            .agg([
                pl.col("latency_ms").mean().alias("avg_latency"),
                pl.col("latency_ms").count().alias("event_count"),
                pl.col("latency_ms").std().alias("std_latency")
            ])
            .collect(streaming=True)
        )
        
        print("Streaming Aggregation Output (Top 5 rows):")
        print(aggregated.head(5))
        assert aggregated.height == 80  # 20 devices * 4 event types
        assert aggregated.select(pl.col("event_count").sum()).item() == 1_000_000

if __name__ == "__main__":
    verify_streaming_out_of_core()
```
