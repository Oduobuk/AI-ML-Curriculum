# Week 1: Graduate-Level Problem Set — Convex Optimization, Regularized Regression & Coordinate Descent

Welcome to the Month 2 Week 1 Problem Set. This suite rigorous covers the mathematics and numerical algorithms of convex linear models: normal equations, singular matrix regularization (Ridge/Tikhonov), subgradient calculus for L1 non-differentiability (Lasso), Coordinate Descent with soft-thresholding, and Elastic Net trade-offs.

---

## Tier 1: Core Mechanics & Analytic Solutions

### Exercise 1: Gauss-Markov Theorem & Normal Equations with Tikhonov Regularization
**Concepts**: Ordinary Least Squares (OLS), Tikhonov regularizer $\lambda I$, matrix conditioning, bias-variance trade-off.

#### Problem Specification
Consider the linear regression objective with $L_2$ regularization:
$$\mathcal{L}(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2$$
where $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, and $\lambda \ge 0$.
1. Analytically derive the gradient $\nabla_w \mathcal{L}(w)$ and set it to zero to prove that:
   $$\hat{w}_{\text{ridge}} = \left(X^T X + n\lambda I_d\right)^{-1} X^T y$$
2. Prove that for any $\lambda > 0$, the matrix $G = X^T X + n\lambda I_d$ is strictly positive definite, guaranteeing invertibility even when $n < d$ or when columns of $X$ are collinear.
3. Express the singular value decomposition of $X = U \Sigma V^T$ and prove that the effective degrees of freedom $\text{df}(\lambda)$ is:
   $$\text{df}(\lambda) = \sum_{j=1}^d \frac{\sigma_j^2}{\sigma_j^2 + n\lambda}$$
4. Implement a Python routine to compute $\text{df}(\lambda)$ across a logarithmic sweep of $\lambda \in [10^{-4}, 10^4]$.

#### Verification Solution
```python
import numpy as np

def compute_effective_degrees_of_freedom(X: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    n, d = X.shape
    # Singular values
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    
    dfs = []
    for lam in lambdas:
        df_lam = np.sum(s2 / (s2 + n * lam))
        dfs.append(df_lam)
    return np.array(dfs)

def solve_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    A = X.T @ X + n * lam * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 100, 20
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    y = X @ true_w + np.random.normal(0, 0.5, size=n)
    
    lambdas = np.logspace(-3, 3, 7)
    dfs = compute_effective_degrees_of_freedom(X, lambdas)
    
    print("Lambda values: ", lambdas)
    print("Degrees of Freedom:", np.round(dfs, 2))
    assert dfs[0] > dfs[-1]
    assert np.isclose(dfs[0], d, atol=0.5)
```

---

### Exercise 2: Subgradient Calculus & Soft-Thresholding Operator
**Concepts**: Subgradient $\partial |w|$, directional derivatives, Lasso non-smooth optimization.

#### Problem Specification
The $L_1$ Lasso penalty introduces non-differentiability at $w_j = 0$:
$$\min_w \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \|w\|_1$$
1. Define the subdifferential $\partial |w_j|$:
   $$\partial |w_j| = \begin{cases} \{1\} & \text{if } w_j > 0 \\ \{-1\} & \text{if } w_j < 0 \\ [-1, 1] & \text{if } w_j = 0 \end{cases}$$
2. Prove that the univariate problem $\min_w \frac{1}{2} (w - z)^2 + \lambda |w|$ has the exact closed-form solution given by the soft-thresholding operator $\mathcal{S}_\lambda(z)$:
   $$\mathcal{S}_\lambda(z) = \text{sign}(z) \max(0, |z| - \lambda)$$
3. Implement `soft_threshold(z, lam)` and verify that it sets values strictly inside $[-\lambda, \lambda]$ to exact zero.

#### Verification Solution
```python
import numpy as np

def soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(z) * np.maximum(0.0, np.abs(z) - lam)

# Verification Run
if __name__ == "__main__":
    z_vals = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    thresholded = soft_threshold(z_vals, lam=1.0)
    print("Input:       ", z_vals)
    print("Thresholded: ", thresholded)
    expected = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    assert np.allclose(thresholded, expected)
```

---

## Tier 2: Coordinate Descent Optimization Engine

### Exercise 3: Coordinate Descent for Lasso from Scratch
**Concepts**: Cyclic coordinate descent, partial residuals, active set sparsity.

#### Problem Specification
1. Implement cyclic Coordinate Descent for Lasso:
   - For feature coordinate $j = 1, \dots, d$:
     - Compute the partial residual: $r_{-j} = y - \sum_{k \neq j} X_{:, k} w_k = (y - Xw) + X_{:, j} w_j$.
     - Compute the unconstrained correlation: $\rho_j = X_{:, j}^T r_{-j}$.
     - Update coordinate: $w_j = \frac{\mathcal{S}_{n\alpha}(\rho_j)}{\|X_{:, j}\|_2^2}$.
2. Run Coordinate Descent on a synthetic dataset where only 5 of 50 features are truly predictive.
3. Verify that the algorithm converges monotonically and forces the 45 irrelevant weights to exact zero ($w_k = 0.0$).

#### Verification Solution
```python
import numpy as np

def coordinate_descent_lasso(X: np.ndarray, y: np.ndarray, alpha: float, max_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d)
    norm_sq = np.sum(X ** 2, axis=0)
    
    for iteration in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            # Compute partial residual dot product: X_j.T @ (y - Xw + X_j * w_j)
            residual = y - X @ w
            rho_j = np.dot(X[:, j], residual) + norm_sq[j] * w[j]
            # Update coordinate using soft-thresholding
            w[j] = np.sign(rho_j) * max(0.0, abs(rho_j) - n * alpha) / norm_sq[j]
            
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 200, 30
    X = np.random.randn(n, d)
    # Only first 3 features have true signal
    true_w = np.zeros(d)
    true_w[0] = 5.0
    true_w[1] = -3.5
    true_w[2] = 2.0
    y = X @ true_w + np.random.normal(0, 0.2, size=n)
    
    fitted_w = coordinate_descent_lasso(X, y, alpha=0.1)
    print("Fitted Weights (Top 5):", np.round(fitted_w[:5], 3))
    print("Zero weights count (out of 30):", np.sum(fitted_w == 0.0))
    assert np.all(fitted_w[3:] == 0.0)
    assert np.all(fitted_w[:3] != 0.0)
```

---

## Tier 3: Logistic Regression with Newton-Raphson (IRLS)

### Exercise 4: Iteratively Reweighted Least Squares (IRLS) for Binary Classification
**Concepts**: Cross-entropy Hessian, Newton-Raphson update, working response vector $z_t$, Fisher scoring.

#### Problem Specification
1. The negative log-likelihood for binary logistic regression is:
   $$\mathcal{L}(w) = -\sum_{i=1}^n \left(y_i \ln p_i + (1 - y_i) \ln(1 - p_i)\right)$$
   where $p_i = \sigma(x_i^T w) = \frac{1}{1 + e^{-x_i^T w}}$.
2. Derive the gradient $\nabla \mathcal{L}(w) = X^T (p - y)$ and the Hessian $H(w) = X^T S X$, where $S = \text{diag}(p_1(1-p_1), \dots, p_n(1-p_n))$.
3. Implement the Iteratively Reweighted Least Squares (IRLS) algorithm:
   $$w_{t+1} = (X^T S_t X)^{-1} X^T S_t z_t \quad \text{where } z_t = X w_t - S_t^{-1}(p_t - y)$$
4. Assert quadratic local convergence to the optimum in under 10 iterations.

#### Verification Solution
```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def irls_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d)
    
    for it in range(max_iter):
        p = sigmoid(X @ w)
        s = p * (1.0 - p) + 1e-5  # Variance weights
        S = np.diag(s)
        
        grad = X.T @ (p - y)
        H = X.T @ S @ X
        
        # Newton step: delta_w = - H^{-1} * grad
        delta_w = np.linalg.solve(H, grad)
        w = w - delta_w
        
        if np.linalg.norm(delta_w) < tol:
            print(f"IRLS converged in {it + 1} iterations.")
            return w
    return w

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 300, 3
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -1.5, 3.0])
    prob = sigmoid(X @ true_w)
    y = (np.random.rand(n) < prob).astype(float)
    
    w_fit = irls_logistic_regression(X, y)
    print("True weights:  ", true_w)
    print("Fitted weights:", np.round(w_fit, 3))
    assert np.allclose(w_fit, true_w, atol=0.5)
```
