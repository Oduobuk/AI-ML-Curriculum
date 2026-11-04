# Week 1 Production Engineering Assignment: High-Throughput Out-of-Core Parquet Feature Pipeline

## Executive Overview & Architectural Motivation
Enterprise ML feature engineering pipelines ingest hundreds of gigabytes of unstructured or semi-structured raw event logs every hour. Legacy systems written in unoptimized Pandas frequently suffer out-of-memory (`OOMKilled`) termination, poor multi-core CPU utilization, and massive serialization overheads.

In this assignment, you will architect an industrial-grade **High-Throughput Out-of-Core Feature Pipeline** using Polars, Apache Arrow, and Parquet.

Your system will ingest streaming partitioned datasets, perform zero-copy feature extraction, construct temporal rolling aggregation windows, compute statistical normalization statistics, and output clean, columnar partitioned Parquet training splits ready for distributed training.

---

## Technical Specifications & Architecture Diagram

```
                 Incoming Multi-GB Partitioned Parquet Logs
                 [/data/raw/year=2026/month=03/day=*/part-*.parquet]
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          Polars LazyFrame Scanner         │
                  │  (Predicate + Projection Pushdown Filter) │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │      Vectorized Feature Engineering       │
                  │   - Outlier Clipping via Dynamic Medians  │
                  │   - Temporal Rolling Window Features      │
                  │   - Z-score & Robust MinMax Scalers       │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   Out-of-Core Streaming Sink Pipeline     │
                  │         (.collect(streaming=True))        │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  Clean Engineered Partitions in Memory-Mapped Arrow
                  [/data/engineered/train/part-*.parquet]
```

---

## Detailed Deliverable Requirements

### 1. Ingestion & Lazy Schema Enforcement (`ingest.py`)
- Implement `LazyFeaturePipeline` initialized with a source directory of multi-file partitioned Parquet datasets.
- Enforce strict Apache Arrow schema contracts:
  - Cast string identifiers into `pl.Categorical`.
  - Parse ISO-8601 strings into microsecond timestamps `pl.Datetime("us")`.
  - Enforce explicit precision on float metrics (`pl.Float32` for neural network compatibility).

### 2. Zero-Copy Feature Transformations (`features.py`)
Implement vectorized feature engineering pipelines using pure Polars expressions:
1. **Dynamic Log Ratios**:
   $$\text{log\_latency\_ratio} = \log\left(1.0 + \frac{\text{latency}}{\text{median\_latency}_{\text{device}}}\right)$$
2. **Rolling Temporal Windows**:
   - Compute moving average, standard deviation, and exponential moving average (EMA) of transaction volume across sliding windows (`5m`, `1h`).
3. **Robust Standardization**:
   $$z = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
   where $\text{IQR}(x) = Q_{0.75}(x) - Q_{0.25}(x)$.

### 3. Out-of-Core Streaming Export (`export.py`)
- Stream results into partitioned Parquet sinks using `.sink_parquet(..., compression="zstd")`.
- Assert that peak memory footprint during transformation remains bounded (under 500MB RAM even for multi-GB input scales).

---

## Starter Code Template & Submission Skeleton

```python
"""
high_throughput_feature_pipeline.py - Production Polars Feature Engine
"""
import polars as pl
import numpy as np
import tempfile
import os
import time
from dataclasses import dataclass
from typing import List

@dataclass
class PipelineConfig:
    input_path: str
    output_path: str
    zstd_compression_level: int = 3
    streaming_mode: bool = True

class EnterpriseFeaturePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build_feature_graph(self) -> pl.LazyFrame:
        """Constructs unexecuted LazyFrame query graph with pushdown optimizations."""
        lf = pl.scan_parquet(os.path.join(self.config.input_path, "*.parquet"))
        
        # 1. Type casting and sanitization
        sanitized = lf.with_columns([
            pl.col("device_id").cast(pl.Categorical),
            pl.col("latency_ms").cast(pl.Float32),
            pl.col("error_count").cast(pl.UInt16)
        ])
        
        # 2. Windowed & Aggregated Transformations
        engineered = sanitized.with_columns([
            # Device-level median latency
            pl.col("latency_ms").median().over("device_id").alias("device_median_latency"),
            # Rolling transaction rate
            pl.col("latency_ms").rolling_mean(window_size=100).over("device_id").alias("rolling_mean_latency"),
            pl.col("latency_ms").rolling_std(window_size=100).over("device_id").alias("rolling_std_latency"),
        ]).with_columns([
            # Robust Z-score feature
            ((pl.col("latency_ms") - pl.col("device_median_latency")) / 
             (pl.col("rolling_std_latency") + 1e-5)).alias("robust_latency_zscore")
        ])
        
        return engineered

    def execute_streaming_pipeline(self) -> float:
        """Executes feature engineering and sinks directly to disk with zero-copy stream."""
        t0 = time.perf_counter()
        graph = self.build_feature_graph()
        
        output_file = os.path.join(self.config.output_path, "engineered_features.parquet")
        # In Polars, sink_parquet streams lazy evaluation directly to disk
        graph.sink_parquet(output_file, compression="zstd", compression_level=self.config.zstd_compression_level)
        return time.perf_counter() - t0

# Verification Run
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        # Generate 500,000 synthetic log records across 5 parquet partitions
        print("Generating mock partitioned parquet datasets...")
        for p in range(5):
            df_part = pl.DataFrame({
                "timestamp": np.arange(100_000, dtype=np.int64),
                "device_id": np.random.choice(["edge_gateway_1", "edge_gateway_2", "edge_gateway_3"], size=100_000),
                "latency_ms": np.random.exponential(scale=30.0, size=100_000),
                "error_count": np.random.poisson(lam=0.2, size=100_000)
            })
            df_part.write_parquet(os.path.join(tmp_in, f"partition_{p}.parquet"))
            
        config = PipelineConfig(input_path=tmp_in, output_path=tmp_out)
        pipeline = EnterpriseFeaturePipeline(config)
        
        elapsed = pipeline.execute_streaming_pipeline()
        print(f"Pipeline executed in {elapsed:.3f} seconds.")
        
        # Verify output integrity
        result_df = pl.read_parquet(os.path.join(tmp_out, "engineered_features.parquet"))
        print(f"Engineered Dataset Shape: {result_df.shape}")
        print("Columns created:", result_df.columns)
        assert "robust_latency_zscore" in result_df.columns
        assert result_df.height == 500_000
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Arrow & LazyFrame Optimization** | 30 | Fully lazy execution graph verified with `.explain()`; no eager computations before sink. |
| **Vectorized Feature Engineering** | 25 | All transformations implemented purely in Polars expressions with zero Python `.apply()` loops. |
| **Out-of-Core Streaming Sink** | 25 | Successful disk streaming using `.sink_parquet()`; memory consumption strictly bounded. |
| **Schema Validation & Typing** | 10 | Exact categorical encoding, float32 precision, and robust null imputation. |
| **Performance Benchmark & Quality** | 10 | Execution completes under 5 seconds for 500k rows with reproducible test suite. |
