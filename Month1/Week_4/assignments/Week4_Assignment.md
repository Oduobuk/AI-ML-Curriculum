# Week 4 Production Engineering Assignment: Automated Feature Drift & Data Profiling Dashboard Suite

## Executive Overview & Architectural Motivation
In real-world ML deployment pipelines, models do not degrade because of internal code bugs; they degrade because the production data environment drifts away from the training distribution. Silent data corruptions, missing upstream schema columns, covariate shifts, and concept drift erode inference reliability without raising runtime exceptions.

In this assignment, you will architect an **Automated Feature Drift & Data Profiling Suite** integrating **Polars**, statistical divergence scoring (Population Stability Index & Kolmogorov-Smirnov tests), and an interactive **Streamlit** telemetry dashboard.

Your suite will continuously monitor live inference streams, compare them against frozen training baselines, generate structured drift alert reports, and visually pinpoint affected feature coordinates.

---

## Technical Specifications & Architecture Diagram

```
                 Training Baseline Dataset        Live Production Stream
                 [baseline_features.parquet]     [live_inference_stream]
                              │                             │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     Statistical Drift Engine │
                              │   - Population Stability (PSI)│
                              │   - 2-Sample KS-Test (p-vals)│
                              │   - Vectorized Polars Aggs  │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    Automated Drift Alerting │
                              │   - Green: PSI < 0.10       │
                              │   - Yellow: 0.10 <= PSI <= 0.25│
                              │   - Red Alert: PSI > 0.25   │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Streamlit Interactive Visual │
                              │   - Distribution Overlays   │
                              │   - Correlation Difference   │
                              └─────────────────────────────┘
```

---

## Detailed Deliverable Requirements

### 1. Vectorized Drift Calculation Engine (`drift_engine.py`)
Implement `StatisticalDriftDetector`:
- Accepts `baseline_df: pl.DataFrame` and `current_df: pl.DataFrame`.
- **Population Stability Index (PSI)**: Vectorized quantile binning with numerical smoothing $\epsilon = 10^{-4}$.
- **Two-Sample Kolmogorov-Smirnov (KS) Test**:
  $$D = \sup_x |F_{\text{baseline}}(x) - F_{\text{current}}(x)|$$
  Computes empirical $D$-statistic and associated $p$-value for all numerical features.
- Categorical Drift: Chi-square test of independence or total variation distance.

### 2. Automated Diagnostic Report Generation (`reporter.py`)
- Emits a machine-readable JSON alert summary:
  ```json
  {
    "timestamp": "2026-03-24T12:00:00Z",
    "overall_status": "WARNING",
    "features_monitored": 24,
    "drifted_features": ["device_latency", "battery_temp"],
    "details": {
      "device_latency": {"psi": 0.284, "ks_stat": 0.142, "p_value": 0.0001, "status": "CRITICAL"}
    }
  }
  ```

### 3. Interactive Telemetry Dashboard (`dashboard.py`)
Construct a complete Streamlit application:
- File uploader or automated path picker for baseline vs current Parquet datasets.
- Top-level KPI cards displaying Monitored Features, Drift Alerts, and Maximum Observed PSI.
- Interactive histogram/KDE distribution overlay comparing baseline vs target for selected features.
- Difference heatmap visualizing changes in Pearson/Spearman correlation matrices:
  $$\Delta R = R_{\text{current}} - R_{\text{baseline}}$$

---

## Starter Code Template & Submission Skeleton

```python
"""
drift_monitor.py - Production Drift Detection Suite
"""
import polars as pl
import numpy as np
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class DriftReport:
    feature_name: str
    psi: float
    ks_statistic: float
    p_value: float
    status: str

class StatisticalDriftDetector:
    def __init__(self, psi_threshold_warning: float = 0.10, psi_threshold_critical: float = 0.25):
        self.psi_warn = psi_threshold_warning
        self.psi_crit = psi_threshold_critical

    def compute_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        P = np.clip(base_counts / len(baseline), 1e-4, 1.0)
        Q = np.clip(curr_counts / len(current), 1e-4, 1.0)
        P /= np.sum(P)
        Q /= np.sum(Q)
        
        return float(np.sum((Q - P) * np.log(Q / P)))

    def audit_features(self, baseline_df: pl.DataFrame, current_df: pl.DataFrame) -> List[DriftReport]:
        numeric_cols = [c for c, dt in zip(baseline_df.columns, baseline_df.dtypes) if dt.is_numeric()]
        reports = []
        
        for col in numeric_cols:
            base_vals = baseline_df.select(pl.col(col).drop_nulls()).to_numpy().flatten()
            curr_vals = current_df.select(pl.col(col).drop_nulls()).to_numpy().flatten()
            
            psi = self.compute_psi(base_vals, curr_vals)
            ks_res = ks_2samp(base_vals, curr_vals)
            
            if psi >= self.psi_crit:
                status = "CRITICAL"
            elif psi >= self.psi_warn:
                status = "WARNING"
            else:
                status = "HEALTHY"
                
            reports.append(DriftReport(
                feature_name=col,
                psi=psi,
                ks_statistic=float(ks_res.statistic),
                p_value=float(ks_res.pvalue),
                status=status
            ))
            
        return reports

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n = 20_000
    
    # Baseline
    df_base = pl.DataFrame({
        "feature_stable": np.random.normal(0, 1, size=n),
        "feature_drifted": np.random.normal(5, 2, size=n)
    })
    
    # Current (feature_drifted has shifted mean and variance)
    df_curr = pl.DataFrame({
        "feature_stable": np.random.normal(0, 1, size=n),
        "feature_drifted": np.random.normal(7, 3.5, size=n)
    })
    
    detector = StatisticalDriftDetector()
    results = detector.audit_features(df_base, df_curr)
    
    for r in results:
        print(f"Feature: {r.feature_name:<16} | PSI: {r.psi:.4f} | KS p-val: {r.p_value:.2e} | Status: {r.status}")
        
    assert results[0].status == "HEALTHY"
    assert results[1].status == "CRITICAL"
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Statistical Rigor (PSI & KS-Test)** | 35 | Flawless mathematical formulation, quantile stability, robust epsilon smoothing. |
| **Vectorized Profiling & Performance** | 25 | Fully vectorized Polars operations running sub-second over large record batches. |
| **Automated Alerting & JSON Reporting** | 20 | Clean structured report schemas with warning/critical threshold classifications. |
| **Interactive Dashboard & Visualization** | 10 | Well-structured Streamlit dashboard displaying KDE distributions and correlation diffs. |
| **Verification & Test Coverage** | 10 | Automated test suite verifying detection of calibrated artificial shifts. |
