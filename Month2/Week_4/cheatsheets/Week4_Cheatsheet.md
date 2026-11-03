# Month 2, Week 4 Cheatsheet: Leak-Proof Preprocessing Pipelines, TreeSHAP & ONNX Runtime

---

## 1. Scikit-Learn `ColumnTransformer` Leak-Proof Architecture

Encapsulates all preprocessing into an atomic estimator to prevent train-test data leakage:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

num_cols = ["age", "income", "transaction_amount"]
cat_cols = ["merchant_category", "state"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), num_cols),
        ("cat", TargetEncoder(smooth="auto", cv=5), cat_cols)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

# Fitting strictly computes stats on X_train only!
pipeline.fit(X_train, y_train)
```

---

## 2. Empirical Bayes Target Encoding with $m$-Estimate Smoothing

Replaces high-cardinality categorical levels with smoothed Bayesian prior targets:
$$\hat{S}_c = \frac{n_c \bar{y}_c + m \mu_{\text{global}}}{n_c + m}$$
- $n_c$: Frequency count of category $c$ in training set.
- $\bar{y}_c$: Local empirical target mean for category $c$.
- $\mu_{\text{global}}$: Global target mean across the entire training dataset.
- $m$: Smoothing weight (pseudo-counts). When $n_c \to 0$, $\hat{S}_c \to \mu_{\text{global}}$ (prevents rare category overfitting).

---

## 3. TreeSHAP & Cooperative Game Theory Axioms

SHAP computes additive feature attribution values satisfying four cooperative game theory axioms:
$$f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$

### The 4 Core Shapley Axioms
1. **Efficiency**: $\sum_{j=1}^M \phi_j(x) = f(x) - \mathbb{E}[f(X)]$. The sum of local attributions exactly equals the difference between model output and baseline expected value.
2. **Symmetry**: If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all coalitions $S$, then $\phi_i = \phi_j$.
3. **Dummy Player**: If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity**: For ensemble $f + g$, $\phi_i(f + g) = \phi_i(f) + \phi_i(g)$.

```python
import shap

explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
shap_values = explainer(transformed_features)

# Waterfall plot: Local sample explanation
# Summary plot: Global feature importance
```

---

## 4. Production ONNX Serialization & Sub-Millisecond Serving

Compiles Python models into static computational graphs executed by optimized C++ runtimes:
```python
from skl2onnx import to_onnx
import onnxruntime as ort

# 1. Export Scikit-Learn pipeline to ONNX
onnx_model = to_onnx(pipeline, X_train[:1])
with open("/models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 2. Production Inference with ONNX Runtime
session = ort.InferenceSession("/models/model.onnx")
inputs = {session.get_inputs()[0].name: sample_data}
probabilities = session.run(None, inputs)[1]  # Sub-millisecond latency
```
