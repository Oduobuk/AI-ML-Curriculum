# Week 4: Graduate-Level Problem Set — Exploratory Data Analysis, Feature Drift & Statistical Profiling

Welcome to the Month 1 Week 4 Problem Set. This suite covers the rigorous analytical techniques required to audit dataset quality, detect covariate shift, compute multi-variable correlations, and engineer interactive visual dashboards for production ML models.

---

## Tier 1: Core Statistical Distributions & Bivariate Profiling

### Exercise 1: Population Stability Index (PSI) & Covariate Shift Quantification
**Concepts**: Population Stability Index, Kullback-Leibler divergence, covariate shift, probability binning.

#### Problem Specification
In production ML systems, feature drift occurs when the distribution of an incoming feature $Q(x)$ departs from the baseline training distribution $P(x)$.
1. Formulate the Population Stability Index (PSI) across $K$ discrete quantile bins:
   $$\text{PSI} = \sum_{k=1}^K \left(Q_k - P_k\right) \times \ln\left(\frac{Q_k}{P_k}\right)$$
   where $P_k$ and $Q_k$ are the fraction of observations falling into bin $k$.
2. Implement `calculate_psi(baseline, target, bins=10, epsilon=1e-4)`.
3. Generate a baseline normal distribution $\mathcal{N}(0, 1)$ and evaluate PSI under:
   - No drift: $\mathcal{N}(0, 1)$
   - Slight drift: $\mathcal{N}(0.1, 1.05)$
   - Significant drift: $\mathcal{N}(0.5, 1.5)$
4. Benchmark empirical results against industry alert thresholds ($\text{PSI} < 0.1 \implies \text{Stable}$, $0.1 \le \text{PSI} \le 0.25 \implies \text{Moderate Drift}$, $\text{PSI} > 0.25 \implies \text{Severe Drift}$).

#### Verification Solution
```python
import numpy as np

def calculate_psi(baseline: np.ndarray, target: np.ndarray, bins: int = 10, epsilon: float = 1e-4) -> float:
    # 1. Establish quantile boundaries on baseline
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # 2. Count proportions
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)
    
    P = baseline_counts / len(baseline)
    Q = target_counts / len(target)
    
    # Smooth zero probabilities
    P = np.clip(P, epsilon, 1.0)
    Q = np.clip(Q, epsilon, 1.0)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # 3. Sum PSI formula
    psi_value = np.sum((Q - P) * np.log(Q / P))
    return float(psi_value)

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=50_000)
    stable_target = np.random.normal(loc=0.0, scale=1.0, size=50_000)
    slight_drift = np.random.normal(loc=0.15, scale=1.05, size=50_000)
    severe_drift = np.random.normal(loc=0.6, scale=1.4, size=50_000)
    
    psi_stable = calculate_psi(baseline, stable_target)
    psi_slight = calculate_psi(baseline, slight_drift)
    psi_severe = calculate_psi(baseline, severe_drift)
    
    print(f"PSI (Stable):  {psi_stable:.4f}")
    print(f"PSI (Slight):  {psi_slight:.4f}")
    print(f"PSI (Severe):  {psi_severe:.4f}")
    
    assert psi_stable < 0.05
    assert 0.05 < psi_slight < 0.25
    assert psi_severe > 0.25
```

---

### Exercise 2: Non-Linear Correlation Analysis: Distance Correlation & Mutual Information
**Concepts**: Pearson correlation limitations, Distance Correlation, Mutual Information, non-linear dependencies.

#### Problem Specification
Pearson's $r$ measures strictly linear association and yields $r \approx 0$ for non-linear deterministic functions (e.g., $y = x^2$ on $[-1, 1]$):
1. Implement empirical Mutual Information $I(X; Y)$ or Distance Correlation for continuous variables.
2. Generate $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2 + \epsilon$.
3. Compute:
   - Pearson's correlation coefficient $r$.
   - Mutual Information estimate via Scikit-Learn `mutual_info_regression`.
4. Demonstrate why classical correlation fails to alert data engineers to non-linear feature interactions.

#### Verification Solution
```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

def evaluate_nonlinear_dependencies():
    np.random.seed(42)
    n = 10_000
    x = np.random.uniform(-1.0, 1.0, size=n)
    y = x**2 + np.random.normal(0, 0.02, size=n)
    
    # 1. Pearson Linear Correlation
    r, p_val = pearsonr(x, y)
    
    # 2. Mutual Information
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
    
    print(f"Pearson Correlation r: {r:.4f} (p-value: {p_val:.4f})")
    print(f"Mutual Information:    {mi:.4f}")
    
    # Pearson indicates no relationship (|r| < 0.05), whereas MI detects high dependency (> 0.5)
    assert abs(r) < 0.05
    assert mi > 0.5

if __name__ == "__main__":
    evaluate_nonlinear_dependencies()
```

---

## Tier 2: Vectorized EDA Profiling Engine

### Exercise 3: High-Performance Automated Metadata Profiler in Polars
**Concepts**: Quantile summaries, skewness, kurtosis, missing value cardinality, memory allocation profiling.

#### Problem Specification
Build a high-performance feature profiling class `DataProfiler` in Polars:
1. Accepts any `pl.DataFrame`.
2. Computes per-column summary metrics in a single vectorized pass:
   - Data type and null percentage.
   - Distinct unique value count.
   - Mean, standard deviation, median, IQR ($Q_{0.75} - Q_{0.25}$).
   - Skewness ($g_1 = \frac{m_3}{m_2^{3/2}}$).
3. Executes across 1,000,000 rows in less than 250 milliseconds.

#### Verification Solution
```python
import polars as pl
import numpy as np
import time

def profile_dataset(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.perf_counter()
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype.is_numeric()]
    
    exprs = []
    for col in numeric_cols:
        exprs.extend([
            pl.col(col).null_count().alias(f"{col}__nulls"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).quantile(0.75).alias(f"{col}__q75"),
            pl.col(col).quantile(0.25).alias(f"{col}__q25"),
            pl.col(col).skew().alias(f"{col}__skew")
        ])
        
    summary = df.select(exprs)
    elapsed = time.perf_counter() - t0
    print(f"Vectorized EDA profiling on {df.height:,} records took {elapsed*1000:.2f} ms")
    return summary

# Verification Run
if __name__ == "__main__":
    n = 1_000_000
    df = pl.DataFrame({
        "feature_1": np.random.normal(0, 1, size=n),
        "feature_2": np.random.exponential(scale=2.0, size=n),
        "feature_3": np.random.uniform(10, 50, size=n)
    })
    
    summary = profile_dataset(df)
    assert summary.width == 18
```
