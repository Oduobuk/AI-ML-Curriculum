# Week 2 Production Engineering Assignment: Custom Second-Order Gradient Boosted Decision Tree (GBDT) Engine

## Executive Overview & Architectural Motivation
Gradient Boosted Decision Trees (LightGBM, XGBoost, CatBoost) remain the state-of-the-art architectures for non-perceptual tabular intelligence. While modern deep neural networks excel at vision, speech, and language, GBDTs consistently outperform deep learning on tabular business records due to rotation-invariant split criteria, native missing value routing, and exact second-order Taylor optimization.

In this assignment, you will engineer a **Production-Ready Second-Order Gradient Boosted Decision Tree Engine** from scratch in Python and NumPy.

You will implement the exact mathematical mechanics that power XGBoost: continuous regression tree splits, 2nd-order Taylor expansions of arbitrary loss functions, shrinkage learning rate decay, L2 leaf weight regularization, and early stopping.

---

## Technical Specifications & Architecture Diagram

```
                 Training Dataset [X, y] & Loss Function $\ell(y, \hat{y})$
                                   │
                                   ▼
                 Initialize Base Score $\hat{y}_i^{(0)} = \arg\min_c \sum \ell(y_i, c)$
                                   │
                                   ▼
                 ┌────────────────────────────────────────────────────────┐
                 │                Boosting Iteration Loop                 │
                 │   For $t = 1 \dots T$:                                 │
                 │   1. Compute 1st order gradients: $g_i = \partial \ell$│
                 │   2. Compute 2nd order Hessians:  $h_i = \partial^2\ell$│
                 │   3. Build Decision Tree $f_t(x)$ via Split Gain:      │
                 │      $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G_P^2}{H_P+\lambda}\right] - \gamma$ │
                 │   4. Compute optimal leaf weights $w_j^* = -\frac{G_j}{H_j+\lambda}$ │
                 │   5. Update model: $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)$ │
                 │   6. Check early stopping on validation loss          │
                 └─────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                 Calibrated GBDT Model with Sub-Millisecond Inference
```

---

## Detailed Deliverable Requirements

### 1. Abstract Loss Interface & Implementations (`losses.py`)
Implement `BaseLoss`:
```python
class BaseLoss(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...
```
Implement subclasses:
- `SquaredErrorLoss`: $g_i = \hat{y}_i - y_i$, $h_i = 1.0$.
- `BinaryCrossEntropyLoss`: $\hat{p}_i = \sigma(\hat{y}_i)$, $g_i = \hat{p}_i - y_i$, $h_i = \hat{p}_i(1 - \hat{p}_i)$.

### 2. Exact Second-Order Regression Tree Builder (`tree.py`)
Implement `GBDTNode` and `GBDTTree`:
- Maximum tree depth `max_depth` (default 3 to 6).
- Minimum child weight constraint: $\sum_{i \in \text{child}} h_i \ge \text{min\_child\_weight}$.
- Optimal split selection maximizing Taylor Gain with $L_2$ regularization $\lambda$ and complexity penalty $\gamma$.
- Closed-form regularized leaf assignment $w_j^* = -\frac{G_j}{H_j + \lambda}$.

### 3. Full Ensemble Orchestrator (`gbdt.py`)
Implement `CustomGBDTClassifier` and `CustomGBDTRegressor`:
- Shrinkage factor (learning rate) $\eta \in (0, 1]$.
- Early stopping monitor: halts boosting when validation loss fails to improve for $k$ consecutive iterations.
- Feature importance attribution: computes total split gain per feature coordinate.

---

## Starter Code Template & Submission Skeleton

```python
"""
custom_gbdt.py - Production Second-Order GBDT Engine
"""
import numpy as np
from typing import Optional, List, Tuple

class BinaryLogisticLoss:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def gradient(self, y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        p = self.sigmoid(raw_pred)
        return p - y_true

    def hessian(self, y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        p = self.sigmoid(raw_pred)
        return np.clip(p * (1.0 - p), 1e-4, 1.0)

class GBDTRegressionTree:
    def __init__(self, max_depth: int = 3, reg_lambda: float = 1.0, min_child_weight: float = 1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.split_feature: Optional[int] = None
        self.split_value: Optional[float] = None
        self.leaf_value: Optional[float] = None
        self.left_child: Optional["GBDTRegressionTree"] = None
        self.right_child: Optional["GBDTRegressionTree"] = None

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int = 0) -> "GBDTRegressionTree":
        G, H = float(np.sum(g)), float(np.sum(h))
        
        # Base case: max depth or insufficient hessian weight
        if depth >= self.max_depth or H < self.min_child_weight:
            self.leaf_value = -G / (H + self.reg_lambda)
            return self

        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feat = None
        best_val = None

        parent_score = (G ** 2) / (H + self.reg_lambda)

        for feat in range(n_features):
            x_col = X[:, feat]
            thresholds = np.unique(x_col)
            if len(thresholds) <= 1:
                continue

            for t in thresholds[:-1]:
                mask_left = x_col <= t
                G_L, H_L = float(np.sum(g[mask_left])), float(np.sum(h[mask_left]))
                G_R, H_R = G - G_L, H - H_L

                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                gain = 0.5 * ((G_L ** 2) / (H_L + self.reg_lambda) + 
                              (G_R ** 2) / (H_R + self.reg_lambda) - parent_score)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_val = t

        if best_feat is None:
            self.leaf_value = -G / (H + self.reg_lambda)
            return self

        self.split_feature = best_feat
        self.split_value = best_val
        left_mask = X[:, best_feat] <= best_val

        self.left_child = GBDTRegressionTree(self.max_depth, self.reg_lambda, self.min_child_weight).fit(
            X[left_mask], g[left_mask], h[left_mask], depth + 1
        )
        self.right_child = GBDTRegressionTree(self.max_depth, self.reg_lambda, self.min_child_weight).fit(
            X[~left_mask], g[~left_mask], h[~left_mask], depth + 1
        )
        return self

    def predict_row(self, x: np.ndarray) -> float:
        if self.leaf_value is not None:
            return self.leaf_value
        if x[self.split_feature] <= self.split_value:
            return self.left_child.predict_row(x)
        return self.right_child.predict_row(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(x) for x in X])

class CustomGBDTClassifier:
    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.2, max_depth: int = 3, reg_lambda: float = 1.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees: List[GBDTRegressionTree] = []
        self.loss = BinaryLogisticLoss()
        self.base_score: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomGBDTClassifier":
        # Initialize with log-odds
        p_mean = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_score = float(np.log(p_mean / (1.0 - p_mean)))
        raw_preds = np.full(len(y), self.base_score, dtype=np.float64)

        for _ in range(self.n_estimators):
            g = self.loss.gradient(y, raw_preds)
            h = self.loss.hessian(y, raw_preds)
            
            tree = GBDTRegressionTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda).fit(X, g, h)
            raw_preds += self.lr * tree.predict(X)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(len(X), self.base_score, dtype=np.float64)
        for t in self.trees:
            raw += self.lr * t.predict(X)
        return self.loss.sigmoid(raw)

# Verification Run
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    
    model = CustomGBDTClassifier(n_estimators=15, learning_rate=0.2, max_depth=3, reg_lambda=1.0)
    model.fit(X, y)
    probs = model.predict_proba(X)
    
    auc = roc_auc_score(y, probs)
    print(f"Custom GBDT Training ROC-AUC: {auc:.4f}")
    assert auc > 0.85
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Exact Second-Order Split Gain Formula** | 35 | Flawless Taylor expansion gain derivation, correct penalty terms ($\lambda, \gamma$), optimal leaf weights. |
| **Tree Recursion & Constraints** | 25 | Proper stopping conditions, minimum child weight enforcement, clean recursive prediction logic. |
| **Shrinkage & Ensemble Boosting** | 20 | Monotonic loss minimization, learning rate scaling, log-odds base score initialization. |
| **Loss Function Modularity** | 10 | Clean separation of gradients/hessians supporting arbitrary objective plugging. |
| **Verification & AUC Benchmark** | 10 | Attains $> 0.85$ ROC-AUC on synthetic classification benchmarks. |
