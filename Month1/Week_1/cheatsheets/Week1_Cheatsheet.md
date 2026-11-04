# Month 1, Week 1 Cheatsheet: High-Performance Data Processing with Polars & Apache Arrow

---

## 1. Apache Arrow Columnar Memory Model vs Pandas

```
                      Row-Oriented Layout (Pandas / NumPy)
 ┌─────────┬─────────┬─────────┐ ┌─────────┬─────────┬─────────┐
 │ Row 0:  │ val_A   │ val_B   │ │ Row 1:  │ val_A   │ val_B   │ ...
 └─────────┴─────────┴─────────┘ └─────────┴─────────┴─────────┘
 (Cache thrashing during column aggregations; requires full row loads)

                      Columnar Layout (Apache Arrow / Polars)
 ┌─────────────────────────────────────────────────────────────┐
 │ Column A Buffer (contiguous 32-bit floats in L1/L2 cache)   │
 ├─────────────────────────────────────────────────────────────┤
 │ Column B Buffer (contiguous 64-bit integers)                │
 ├─────────────────────────────────────────────────────────────┤
 │ Validity Bitmap (1 bit per row: 0 = null, 1 = valid)        │
 └─────────────────────────────────────────────────────────────┘
 (Vectorized SIMD instruction execution: AVX-512 / ARM Neon)
```

### Memory Footprint Comparison
- **Pandas**: Object columns store arrays of 64-bit pointers to heap `PyObject` strings. Nulls represented as float `NaN`, forcing integer columns to cast to float.
- **Polars**: Native contiguous byte buffers. 1-bit validity bitmap. Strings stored in contiguous memory with integer offset arrays.

---

## 2. Polars Query Optimization Engine (`LazyFrame`)

Always prefer `LazyFrame` over eager `DataFrame` for production pipelines:
```python
import polars as pl

lazy_plan = (
    pl.scan_parquet("/data/telemetry/*.parquet")
    .filter((pl.col("device_id") == "sensor_101") & (pl.col("voltage") > 12.0))
    .select(["timestamp", "voltage", "current"])
)

# Inspect query optimizer AST
print(lazy_plan.explain(optimized=False))  # Raw unoptimized AST
print(lazy_plan.explain(optimized=True))   # Fast pushdown AST
```

### Critical Optimizer Pushdowns
- **Predicate Pushdown**: Evaluates `.filter()` inside the Parquet reader, skipping entire row groups based on Parquet page statistics (min/max metadata) without loading them into RAM.
- **Projection Pushdown**: Drops unselected columns before disk reading begins.
- **Common Subplan Elimination**: Identifies duplicate transformation branches and computes them once.

---

## 3. High-Performance Vectorized Polars Expressions

Never drop into Python loops or `.apply()` (which serializes back into CPython). Use native expressions:

### Temporal & Rolling Window Aggregations
```python
df = df.with_columns([
    # Rolling 100-tick moving average
    pl.col("voltage").rolling_mean(window_size=100).over("device_id").alias("rolling_mean_v"),
    
    # Rolling standard deviation
    pl.col("voltage").rolling_std(window_size=100).over("device_id").alias("rolling_std_v"),
    
    # Shift operations (lead / lag)
    pl.col("voltage").shift(1).over("device_id").alias("prev_voltage")
]).with_columns([
    # Vectorized anomaly flag
    ((pl.col("voltage") - pl.col("rolling_mean_v")).abs() > 3.0 * pl.col("rolling_std_v")).alias("is_outlier")
])
```

### Fast Categorical Casting & String Normalization
```python
df = df.with_columns([
    pl.col("category").cast(pl.Categorical),
    pl.col("metric").cast(pl.Float32),
    pl.col("raw_text").str.strip_chars().str.to_lowercase()
])
```

---

## 4. Out-of-Core Streaming Sink Engine

For datasets exceeding available host RAM:
```python
# Streaming collection into RAM
result = lazy_plan.collect(streaming=True)

# Streaming directly to partitioned Parquet sink on disk
lazy_plan.sink_parquet(
    "/data/sink/output.parquet",
    compression="zstd",
    compression_level=3
)
```

---

## 5. Performance Cheat Table: Pandas vs Polars

| Operation | Pandas (Python runtime) | Polars (Rust Arrow runtime) |
|:---|:---|:---|
| **Multi-column group-by** | Single-threaded GIL bound | Multi-threaded lock-free work-stealing |
| **Parquet Scan** | Full disk load into RAM | Memory-mapped chunk streaming + pushdown |
| **Missing Values** | `NaN` float conversion | Native 1-bit Arrow validity bitmap |
| **Memory Reuse** | Frequent full data copying | Zero-copy Arrow slicing (`slice()`) |
