# Week 1 Production Engineering Assignment: High-Dimensional Regularized Linear Engine from Scratch

## Executive Overview & Architectural Motivation
In production tabular intelligence (credit risk assessment, quantitative finance, ad click-through rate prediction), linear and logistic models remain premier baselines due to their strict convexity, sub-millisecond inference latencies, parameter interpretability, and regulatory compliance requirements.

In this assignment, you will engineer a complete, industrial-grade **Regularized Linear & Logistic Modeling Engine** in pure Python and NumPy. You will build and mathematically calibrate:
1. **Ridge Regression via Cholesky Factorization & SVD Inversion**
2. **Elastic Net Regression via Fast Coordinate Descent**
3. **Regularized Binary Logistic Regression via IRLS (Newton-Raphson)**

Your engine will support automated cross-validation, feature standardization with zero data leakage, and export calibrated weights for production inference.

---

## Technical Specifications & Architecture Diagram

```
                 Raw Training Matrix [X_train, y_train]
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │     Welford Feature Standardizer  │
                 │   (Streaming Mean & Variance Norm)│
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │    Convex Solver Engine Selector  │
                 ├─────────────────┬─────────────────┤
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
          │   Ridge     │   │ ElasticNet  │   │  LogReg     │
          │  (Cholesky) │   │ (CoordDesc) │   │   (IRLS)    │
          └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │   Cross-Validated Penalty Search  │
                 │      (Warm-Restart Alpha Path)    │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 Production Calibrated Inference Weights
```

---

## Detailed Deliverable Requirements

### 1. Robust Ridge Solver via Cholesky Factorization (`ridge.py`)
- Standard matrix inversion via `np.linalg.inv` has $O(d^3)$ complexity and catastrophic numerical inaccuracy for ill-conditioned matrices.
- Implement the Ridge solver using Cholesky Decomposition:
  1. Construct $A = X^T X + n\lambda I_d$.
  2. Compute lower-triangular Cholesky factor $L$ such that $L L^T = A$.
  3. Solve $L z = X^T y$ via forward substitution.
  4. Solve $L^T w = z$ via back substitution.
- Handle singular matrix exceptions gracefully by adding dynamic diagonal jitter ($10^{-6}$).

### 2. Elastic Net with Warm-Restart Coordinate Descent (`elastic_net.py`)
- Objective:
  $$\min_w \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|_2^2$$
  where $\alpha$ is overall penalty strength and $\rho \in [0, 1]$ is the $L_1$ ratio.
- Implement cyclic coordinate descent with the modified soft-thresholding operator:
  $$w_j \leftarrow \frac{\mathcal{S}_{n\alpha\rho}\left(\sum_{i=1}^n x_{i, j} r_i^{(-j)}\right)}{\sum_{i=1}^n x_{i, j}^2 + n\alpha(1-\rho)}$$
- Implement a **Warm-Restart Regularization Path**: iterate across a decreasing grid of $\alpha_1 > \alpha_2 > \dots > \alpha_K$, initializing $w^{(k)}$ with the optimal solution of $w^{(k-1)}$.

### 3. Binary Logistic Regression via IRLS (`logistic.py`)
- Implement `RegularizedLogisticRegression`:
  - $L_2$ penalized negative log-likelihood minimization.
  - Newton-Raphson update step with line-search backtracking to guarantee monotonic loss reduction.
  - Return predicted log-odds, calibrated probabilities, and hard class predictions.

---

## Starter Code Template & Submission Skeleton

```python
"""
convex_linear_models.py - Production Convex Regression Suite
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

class CholeskyRidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CholeskyRidgeRegressor":
        n, d = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        
        X_centered = X - x_mean
        y_centered = y - y_mean
        
        A = X_centered.T @ X_centered + (n * self.alpha) * np.eye(d)
        b = X_centered.T @ y_centered
        
        # Cholesky Decomposition: A = L L^T
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A += 1e-6 * np.eye(d)
            L = np.linalg.cholesky(A)
            
        # Forward substitution: L z = b
        z = np.linalg.solve(L, b)
        # Backward substitution: L^T w = z
        self.coef_ = np.linalg.solve(L.T, z)
        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

class ElasticNetCoordinateDescent:
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray = np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray, w_init: Optional[np.ndarray] = None) -> "ElasticNetCoordinateDescent":
        n, d = X.shape
        w = np.zeros(d) if w_init is None else w_init.copy()
        norm_sq = np.sum(X ** 2, axis=0)
        
        l1_pen = n * self.alpha * self.l1_ratio
        l2_denom = n * self.alpha * (1.0 - self.l1_ratio)
        
        for it in range(self.max_iter):
            w_prev = w.copy()
            for j in range(d):
                res = y - X @ w
                rho_j = np.dot(X[:, j], res) + norm_sq[j] * w[j]
                
                # Soft-threshold update
                w[j] = np.sign(rho_j) * max(0.0, abs(rho_j) - l1_pen) / (norm_sq[j] + l2_denom)
                
            if np.max(np.abs(w - w_prev)) < self.tol:
                break
                
        self.coef_ = w
        return self

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 250, 15
    X = np.random.randn(n, d)
    true_w = np.zeros(d)
    true_w[:3] = [4.0, -2.5, 1.8]
    y = X @ true_w + np.random.normal(0, 0.5, size=n)
    
    ridge = CholeskyRidgeRegressor(alpha=0.1).fit(X, y)
    print("Ridge Coefs (Top 5):", np.round(ridge.coef_[:5], 3))
    
    enet = ElasticNetCoordinateDescent(alpha=0.1, l1_ratio=0.8).fit(X, y)
    print("ElasticNet Coefs (Top 5):", np.round(enet.coef_[:5], 3))
    print("Sparsity (Zero Weights):", np.sum(enet.coef_ == 0.0), f"/ {d}")
    
    assert np.all(enet.coef_[3:] == 0.0)
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Cholesky Factorization & Conditioning** | 30 | Flawless $L L^T$ Cholesky decomposition, forward/back solve, jitter regularization. |
| **Elastic Net Coordinate Descent** | 30 | Correct soft-thresholding derivation, warm-restart optimization, strict sparsity. |
| **Logistic Regression & IRLS** | 20 | Monotonic Newton convergence, stable sigmoid implementation, line-search guard. |
| **Mathematical Derivation & Proofs** | 10 | Complete analytical normal equations and subdifferential proofs in markdown. |
| **Verification & Numerical Accuracy** | 10 | Comparison against Scikit-Learn baselines yielding $< 10^{-4}$ parameter divergence. |
