# Month 2, Week 1 Cheatsheet: Convex Optimization, Regularized Linear Models & Coordinate Descent

---

## 1. Ordinary Least Squares (OLS) vs Ridge vs Lasso vs Elastic Net

```
                         Regularization Geometries
       L1 Lasso ($\|w\|_1 \le t$)                  L2 Ridge ($\|w\|_2^2 \le t$)
              ▲                                        ▲
             / \                                     .---.
            /   \                                  /       \
           /     \                                |         |
       ───◄───────►───                        ────◄─────────►────
           \     /                                |         |
            \   /                                  \       /
             \ /                                     '---'
              ▼                                        ▼
    (Corners hit axes -> Sparse $w_j=0$)      (Smooth -> Continuous shrinkage)
```

### Analytical Formulations & Objectives
| Model | Optimization Objective $\mathcal{L}(w)$ | Analytical Solution / Update Rule |
|:---|:---|:---|
| **OLS** | $\frac{1}{2n} \|Xw - y\|_2^2$ | $\hat{w} = (X^T X)^{-1} X^T y$ |
| **Ridge ($L_2$)** | $\frac{1}{2n} \|Xw - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2$ | $\hat{w} = (X^T X + n\lambda I_d)^{-1} X^T y$ |
| **Lasso ($L_1$)** | $\frac{1}{2n} \|Xw - y\|_2^2 + \alpha \|w\|_1$ | Non-smooth; solved via Coordinate Descent |
| **Elastic Net** | $\frac{1}{2n} \|Xw - y\|_2^2 + \alpha \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|_2^2$ | Cyclic Soft-Threshold Coordinate Descent |

---

## 2. Fast Ridge Solver via Cholesky Factorization ($L L^T$)

Never invert $A = (X^T X + n\lambda I)$ with `np.linalg.inv` ($O(d^3)$, numerically unstable):
```python
import numpy as np

def cholesky_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    A = X.T @ X + (n * lam) * np.eye(d)
    b = X.T @ y
    # Cholesky: A = L L^T
    L = np.linalg.cholesky(A)
    # Forward solve: L z = b
    z = np.linalg.solve(L, b)
    # Back solve: L^T w = z
    w = np.linalg.solve(L.T, z)
    return w
```

---

## 3. Subgradient Calculus & Soft-Thresholding Operator

Subdifferential of $|w|$ at $w = 0$ is $[-1, 1]$. The closed-form minimizer of $\frac{1}{2}(w - z)^2 + \lambda |w|$ is:
$$\mathcal{S}_\lambda(z) = \text{sign}(z) \max(0, |z| - \lambda)$$

### Coordinate Descent Update for Lasso
For feature $j$:
$$r_{-j} = y - \sum_{k \neq j} X_{:, k} w_k = (y - Xw) + X_{:, j} w_j$$
$$\rho_j = X_{:, j}^T r_{-j}, \qquad w_j \leftarrow \frac{\mathcal{S}_{n\alpha}(\rho_j)}{\|X_{:, j}\|_2^2}$$

---

## 4. Logistic Regression & Iteratively Reweighted Least Squares (IRLS)

Cross-entropy loss gradient and Hessian:
$$\nabla \mathcal{L}(w) = X^T (p - y), \qquad H(w) = X^T S X \quad \text{where } S_{i, i} = p_i(1 - p_i)$$

### Newton-Raphson Update (IRLS)
$$w_{t+1} = w_t - (X^T S_t X)^{-1} X^T (p_t - y) = (X^T S_t X)^{-1} X^T S_t z_t$$
where $z_t = X w_t - S_t^{-1} (p_t - y)$ is the adjusted working response.

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
```
