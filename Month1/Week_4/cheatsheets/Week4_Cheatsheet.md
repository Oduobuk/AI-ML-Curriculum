# Month 1, Week 4 Cheatsheet: Exploratory Data Analysis, Feature Drift & Dashboard Architecture

---

## 1. Covariate Shift & Population Stability Index (PSI)

Quantifies statistical divergence between baseline training distribution $P$ and live inference distribution $Q$:
$$\text{PSI} = \sum_{k=1}^K (Q_k - P_k) \times \ln\left(\frac{Q_k}{P_k}\right)$$

### Alert Threshold Matrix
| PSI Value | Classification | Production Operational Action |
|:---|:---|:---|
| **$\text{PSI} < 0.10$** | Insignificant / Stable | Baseline operations normal; no action required. |
| **$0.10 \le \text{PSI} \le 0.25$** | Moderate Drift | Monitor closely; trigger automated telemetry alerts. |
| **$\text{PSI} > 0.25$** | Significant / Severe Drift | **Halt live predictions**; trigger continuous model re-training. |

```python
import numpy as np

def calculate_psi(baseline: np.ndarray, target: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    
    P, _ = np.histogram(baseline, bins=bin_edges)
    Q, _ = np.histogram(target, bins=bin_edges)
    
    p_norm = np.clip(P / len(baseline), eps, 1.0)
    q_norm = np.clip(Q / len(target), eps, 1.0)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    
    return float(np.sum((q_norm - p_norm) * np.log(q_norm / p_norm)))
```

---

## 2. Statistical Hypothesis Tests for Drift

### Two-Sample Kolmogorov-Smirnov (KS) Test
Nonparametric test comparing empirical cumulative distribution functions $F_{\text{base}}(x)$ and $F_{\text{curr}}(x)$:
$$D = \sup_x |F_{\text{base}}(x) - F_{\text{curr}}(x)|$$
```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(baseline_feature, live_feature)
# Reject null hypothesis of identical distributions if p_value < 0.05
```

---

## 3. Non-Linear Correlation & Mutual Information

Pearson's $r$ measures strictly linear dependency ($r \approx 0$ for $y = x^2$):
$$\text{Mutual Information: } I(X; Y) = \iint p(x, y) \ln \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

```python
from sklearn.feature_selection import mutual_info_regression

# Captures arbitrary non-linear and sinusoidal associations
mi_scores = mutual_info_regression(X_features, y_target)
```

---

## 4. Vectorized Dataset Profiling in Polars

Processes 1,000,000+ rows in $< 100\text{ ms}$:
```python
import polars as pl

numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]

profile_summary = df.select([
    pl.col(c).null_count().alias(f"{c}__nulls") for c in numeric_cols
] + [
    pl.col(c).mean().alias(f"{c}__mean") for c in numeric_cols
] + [
    pl.col(c).std().alias(f"{c}__std") for c in numeric_cols
] + [
    pl.col(c).skew().alias(f"{c}__skew") for c in numeric_cols
])
```

---

## 5. Production Streamlit Telemetry Dashboard Pattern

```python
import streamlit as st
import polars as pl

st.set_page_config(page_title="Model Telemetry Monitor", layout="wide")
st.title("Enterprise Feature Drift & Data Quality Monitor")

col1, col2, col3 = st.columns(3)
col1.metric("Features Monitored", "42")
col2.metric("Max Feature PSI", "0.284", delta="Drift Alert", delta_color="inverse")
col3.metric("Throughput", "2,400 req/s")

# Display distribution overlay
feature_selected = st.selectbox("Select Feature for Distribution Overlay", ["latency_ms", "token_count"])
# Render interactive chart...
```
