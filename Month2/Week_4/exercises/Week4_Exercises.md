# Week 4: Graduate-Level Problem Set — Production Preprocessing Pipelines, SHAP & ONNX Deployment

Welcome to the Month 2 Week 4 Problem Set. This suite evaluates your mastery of enterprise machine learning deployment workflows: leak-proof Scikit-Learn `ColumnTransformer` pipelines with target encoding, exact TreeSHAP attribution and Shapley axiomatic proofs, and ONNX graph serialization for sub-millisecond production serving.

---

## Tier 1: Core Mechanics & Shapley Axiomatic Foundations

### Exercise 1: Shapley Value Axioms & Exact Coalition Computation
**Concepts**: Shapley value formula, Efficiency axiom, Symmetry axiom, Dummy/Null player axiom, Additivity axiom.

#### Problem Specification
In cooperative game theory, the Shapley value divides a total coalition payoff $v(S)$ among $M$ features:
$$\phi_j(v) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(M - |S| - 1)!}{M!} \left( v(S \cup \{j\}) - v(S) \right)$$
1. Consider a 3-feature cooperative game ($M=3$) where the value function $v(S)$ represents model predictions:
   - $v(\emptyset) = 0$
   - $v(\{1\}) = 10, v(\{2\}) = 2, v(\{3\}) = 5$
   - $v(\{1, 2\}) = 15, v(\{1, 3\}) = 20, v(\{2, 3\}) = 10$
   - $v(\{1, 2, 3\}) = 30$
2. Manually compute the exact Shapley values $\phi_1, \phi_2, \phi_3$.
3. Prove that the Efficiency Axiom holds: $\sum_{j=1}^3 \phi_j = v(\{1, 2, 3\}) - v(\emptyset) = 30$.
4. Implement a Python function `compute_exact_shapley(v_dict, features)` verifying the result.

#### Verification Solution
```python
import itertools
import math

def compute_exact_shapley(v: dict, features: list) -> dict:
    M = len(features)
    shapley_values = {f: 0.0 for f in features}
    
    for j in features:
        other_features = [f for f in features if f != j]
        for s_len in range(M):
            for S in itertools.combinations(other_features, s_len):
                S_tuple = tuple(sorted(S))
                S_with_j = tuple(sorted(S + (j,)))
                
                v_S = v.get(S_tuple, 0.0)
                v_S_j = v.get(S_with_j, 0.0)
                marginal = v_S_j - v_S
                
                weight = (math.factorial(s_len) * math.factorial(M - s_len - 1)) / math.factorial(M)
                shapley_values[j] += weight * marginal
                
    return shapley_values

# Verification Run
if __name__ == "__main__":
    v_game = {
        (): 0.0,
        (1,): 10.0, (2,): 2.0, (3,): 5.0,
        (1, 2): 15.0, (1, 3): 20.0, (2, 3): 10.0,
        (1, 2, 3): 30.0
    }
    feats = [1, 2, 3]
    phi = compute_exact_shapley(v_game, feats)
    print("Exact Shapley Values:", phi)
    print("Sum of Shapley Values:", sum(phi.values()))
    
    assert math.isclose(sum(phi.values()), 30.0)
    assert phi[1] > phi[2]
```

---

## Tier 2: Leak-Proof Preprocessing & Empirical Bayes Target Encoding

### Exercise 2: Out-of-Fold Target Encoding with m-Estimate Smoothing
**Concepts**: High-cardinality categorical encoding, target leakage, $m$-estimate Bayesian prior.

#### Problem Specification
Directly replacing high-cardinality categorical levels with the training target mean leads to severe overfitting:
1. Implement an out-of-fold `BayesianTargetEncoder` using $K$-Fold cross-validation.
2. For each category $c$, calculate the smoothed empirical Bayes estimate:
   $$\hat{S}_c = \frac{n_c \bar{y}_c + m \mu_{\text{global}}}{n_c + m}$$
   where $n_c$ is category frequency, $\bar{y}_c$ is local target mean, $\mu_{\text{global}}$ is global mean, and $m$ is prior weight.
3. Verify that unseen test categories fall back strictly to $\mu_{\text{global}}$ without triggering runtime errors.

#### Verification Solution
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class BayesianTargetEncoder:
    def __init__(self, m: float = 10.0, n_splits: int = 5):
        self.m = m
        self.n_splits = n_splits
        self.global_mean: float = 0.0
        self.mapping_: dict = {}

    def fit_transform(self, categories: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.global_mean = float(np.mean(y))
        encoded = np.empty(len(categories), dtype=np.float64)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(categories):
            cat_tr, y_tr = categories[train_idx], y[train_idx]
            fold_map = {}
            for c in np.unique(cat_tr):
                mask = cat_tr == c
                n_c = np.sum(mask)
                y_c = np.mean(y_tr[mask])
                fold_map[c] = (n_c * y_c + self.m * self.global_mean) / (n_c + self.m)
                
            val_cats = categories[val_idx]
            encoded[val_idx] = [fold_map.get(c, self.global_mean) for c in val_cats]
            
        # Fit global mapping for inference
        for c in np.unique(categories):
            mask = categories == c
            n_c = np.sum(mask)
            y_c = np.mean(y[mask])
            self.mapping_[c] = (n_c * y_c + self.m * self.global_mean) / (n_c + self.m)
            
        return encoded

    def transform(self, categories: np.ndarray) -> np.ndarray:
        return np.array([self.mapping_.get(c, self.global_mean) for c in categories])

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    cats = np.random.choice(["ZIP_A", "ZIP_B", "ZIP_C"], size=1000)
    y = np.where(cats == "ZIP_A", 1, 0)
    
    encoder = BayesianTargetEncoder(m=5.0)
    encoded_train = encoder.fit_transform(cats, y)
    encoded_test = encoder.transform(np.array(["ZIP_A", "ZIP_NEW"]))
    
    print("Sample encoded train values:", encoded_train[:5])
    print("Encoded test values (Known vs Unseen):", encoded_test)
    assert encoded_test[1] == encoder.global_mean
```

---

## Tier 3: TreeSHAP Attribution & Production ONNX Serialization

### Exercise 3: ONNX Serialization & Sub-Millisecond Inference Benchmarking
**Concepts**: Open Neural Network Exchange (ONNX), ONNX Runtime, static computational graphs, inference latency.

#### Problem Specification
1. Train a Gradient Boosting model (`HistGradientBoostingClassifier` or `RandomForestClassifier`) on 10 tabular features.
2. Export the trained model into a static `.onnx` graph using `skl2onnx`.
3. Ingest the graph into `onnxruntime.InferenceSession`.
4. Benchmark single-sample inference latency:
   - Scikit-Learn `.predict_proba()`
   - ONNX Runtime session `.run()`
5. Assert that ONNX Runtime reduces $P_{99}$ latency by at least $2\times$ while matching predictions to within $10^{-5}$.

#### Verification Solution
```python
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

def benchmark_onnx_export():
    np.random.seed(42)
    X = np.random.randn(2000, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=2000)
    
    clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # 1. Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, 10]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    
    # 2. Ingest into ONNX Runtime
    session = ort.InferenceSession(onx.SerializeToString())
    input_name = session.get_inputs()[0].name
    
    sample = X[:1]
    
    # Benchmark Scikit-Learn
    t0 = time.perf_counter()
    for _ in range(500):
        _ = clf.predict_proba(sample)
    t_sklearn = (time.perf_counter() - t0) / 500
    
    # Benchmark ONNX Runtime
    t0 = time.perf_counter()
    for _ in range(500):
        _ = session.run(None, {input_name: sample})
    t_onnx = (time.perf_counter() - t0) / 500
    
    print(f"Scikit-Learn latency: {t_sklearn * 1000:.3f} ms")
    print(f"ONNX Runtime latency: {t_onnx * 1000:.3f} ms")
    assert t_onnx < t_sklearn
```
