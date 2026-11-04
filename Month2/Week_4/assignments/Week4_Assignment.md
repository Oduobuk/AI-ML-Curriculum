# Week 4 Production Engineering Assignment: End-to-End Leak-Proof Pipeline, TreeSHAP Audit & ONNX Serving

## Executive Overview & Architectural Motivation
Deploying machine learning models into high-throughput production microservices requires eliminating three fatal flaws: training-serving skew caused by disparate preprocessing logic, uninterpretable predictions that violate regulatory standards (e.g. EU AI Act, GDPR right-to-explanation), and sluggish Python interpreter inference overhead.

In this capstone assignment for Month 2, you will architect a complete **End-to-End Leak-Proof ML Production Pipeline** combining:
1. **Scikit-Learn ColumnTransformer Preprocessing Pipeline** with Bayesian target encoding and missing value imputation.
2. **Gradient Boosted Classifier** calibrated for risk scoring.
3. **Exact TreeSHAP Interpretability Audit** calculating local and global feature attribution.
4. **ONNX Graph Compilation & Sub-Millisecond Inference Benchmark** comparing native Python against ONNX Runtime.

---

## Technical Specifications & Architecture Diagram

```
                 Raw Heterogeneous Input Records (Numerical + Categorical)
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │         Scikit-Learn ColumnTransformer                │
                 │   - Median Imputation + RobustScaler (Numeric)        │
                 │   - Bayesian Out-of-Fold Target Encoder (Categorical) │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │       Gradient Boosted Classifier Engine              │
                 │              (HistGradientBoosting)                   │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          TreeSHAP Model Explainability Audit          │
                 │   - Local Waterfall Plot Attributions                 │
                 │   - Global Summary Plot & Feature Interactions        │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          ONNX Graph Serialization & Export            │
                 │     (Static Compute Graph + ONNX Runtime Engine)      │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 Sub-Millisecond Production Inference Endpoint
```

---

## Detailed Deliverable Requirements

### 1. Leak-Proof Composite Pipeline (`pipeline.py`)
- Construct a Scikit-Learn `Pipeline` wrapping a `ColumnTransformer`:
  - Numerical features: `SimpleImputer(strategy="median")` followed by `RobustScaler()`.
  - Categorical features: `TargetEncoder(smooth="auto", cv=5)` or custom `BayesianTargetEncoder`.
  - Estimator: `HistGradientBoostingClassifier(random_state=42)`.
- Verify that fitting the pipeline on training data prevents any validation data statistics from leaking into transformers.

### 2. Exact TreeSHAP Interpretability Suite (`explainability.py`)
- Extract TreeSHAP values using `shap.TreeExplainer(model)`:
  - Generate a local Waterfall explanation for individual high-risk predictions:
    $$f(x) = \mathbb{E}[f(X)] + \sum_{j=1}^M \phi_j(x)$$
  - Calculate global mean absolute SHAP values: $\frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$.
  - Assert that feature attributions satisfy the efficiency axiom ($< 10^{-4}$ divergence from raw logits).

### 3. Production ONNX Graph Export & Benchmarking (`onnx_serving.py`)
- Compile the trained Scikit-Learn pipeline into an optimized `.onnx` model using `skl2onnx`.
- Run an inference latency benchmark over 1,000 requests:
  - Compute latency percentiles ($P_{50}, P_{90}, P_{99}$) for native Python vs ONNX Runtime.
  - Verify that both runtimes produce identical output probabilities ($|\hat{y}_{\text{python}} - \hat{y}_{\text{onnx}}| < 10^{-5}$).

---

## Starter Code Template & Submission Skeleton

```python
"""
production_pipeline_shap_onnx.py - Production Preprocessing & Serving Suite
"""
import numpy as np
import pandas as pd
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from skl2onnx import to_onnx
import onnxruntime as ort

def build_and_evaluate_pipeline():
    np.random.seed(42)
    n = 2000
    
    # Synthetic production tabular records
    data = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n).astype(float),
        "income": np.random.exponential(scale=50_000, size=n),
        "debt_ratio": np.random.uniform(0.05, 0.95, size=n),
        "credit_grade": np.random.choice(["TIER_A", "TIER_B", "TIER_C", "TIER_D"], size=n),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], size=n)
    })
    
    # Inject missing values
    data.loc[np.random.choice(n, size=50), "income"] = np.nan
    
    # Target
    y = ((data["debt_ratio"] > 0.5) & (data["credit_grade"].isin(["TIER_C", "TIER_D"]))).astype(int)
    
    num_cols = ["age", "income", "debt_ratio"]
    cat_cols = ["credit_grade", "home_ownership"]
    
    # 1. Composite Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), num_cols),
            ("cat", TargetEncoder(smooth="auto", cv=5), cat_cols)
        ]
    )
    
    # 2. End-to-End Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=30, random_state=42))
    ])
    
    full_pipeline.fit(data, y)
    print("Pipeline successfully fitted without data leakage.")
    
    # 3. Predict on sample
    sample = data.iloc[:1]
    prob_sklearn = full_pipeline.predict_proba(sample)[:, 1][0]
    print(f"Scikit-Learn Inference Probability: {prob_sklearn:.4f}")
    
    # 4. Export to ONNX
    onx_model = to_onnx(full_pipeline, data[:1])
    session = ort.InferenceSession(onx_model.SerializeToString())
    
    # Benchmark ONNX Runtime
    input_dict = {col: data[col].iloc[:1].to_numpy().reshape(-1, 1) for col in data.columns}
    
    # Feed into ONNX
    t0 = time.perf_counter()
    for _ in range(200):
        _ = session.run(None, input_dict)
    t_onnx = (time.perf_counter() - t0) / 200
    
    print(f"ONNX Serving latency per request: {t_onnx * 1000:.3f} ms")
    assert t_onnx < 0.005  # Sub-5ms

if __name__ == "__main__":
    build_and_evaluate_pipeline()
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Leak-Proof Preprocessing Design** | 30 | Flawless `ColumnTransformer` encapsulation, target encoding out-of-fold cross-validation. |
| **TreeSHAP Interpretability & Axioms** | 25 | Correct efficiency verification, local waterfall attributions, global feature ranking. |
| **ONNX Compilation & Serving** | 25 | Successful graph serialization via `to_onnx`, numerical parity validation ($< 10^{-5}$). |
| **Latency Benchmarking** | 10 | Detailed $P_{50}, P_{95}, P_{99}$ comparison proving significant latency reduction. |
| **Verification & Test Coverage** | 10 | End-to-end automated test harness with deterministic seeding. |
