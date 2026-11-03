# Week 3: Graduate-Level Problem Set — Linear Algebra & Matrix Calculus for ML

Welcome to the Week 3 Problem Set. This problem suite transitions from naive arithmetic to the foundational linear algebraic machinery of modern deep learning and machine learning: orthogonal projections, spectral decomposition, singular value decomposition (SVD), Moore-Penrose pseudo-inverses, and matrix differential calculus.

---

## Tier 1: Core Mechanics & Vector Space Geometry

### Exercise 1: Gram-Schmidt Orthogonalization & QR Factorization
**Concepts**: Orthogonality, inner products, projection operators, numerical stability.

#### Problem Specification
Given an linearly independent set of column vectors in matrix $A \in \mathbb{R}^{m \times n}$ where $m \ge n$:
1. Implement Classical Gram-Schmidt (CGS) to produce an orthonormal basis $Q \in \mathbb{R}^{m \times n}$ and upper-triangular matrix $R \in \mathbb{R}^{n \times n}$.
2. Implement Modified Gram-Schmidt (MGS), which orthogonalizes against the newly computed orthonormal vector immediately rather than the original vectors, eliminating numerical catastrophic cancellation.
3. Compare the loss of orthogonality $\|Q^T Q - I_n\|_F$ between CGS and MGS on an ill-conditioned Hilbert-style matrix.

#### Verification Solution
```python
import numpy as np

def classical_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    
    for j in range(n):
        v = A[:, j].astype(np.float64)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-12:
            raise ValueError("Linearly dependent columns detected.")
        Q[:, j] = v / R[j, j]
    return Q, R

def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    V = A.copy().astype(np.float64)
    
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 1e-12:
            raise ValueError("Linearly dependent columns detected.")
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    # Ill-conditioned column vectors
    A = np.array([
        [1.0, 1.0, 1.0],
        [1e-4, 0.0, 0.0],
        [0.0, 1e-4, 0.0],
        [0.0, 0.0, 1e-4]
    ])
    
    Q_cgs, R_cgs = classical_gram_schmidt(A)
    Q_mgs, R_mgs = modified_gram_schmidt(A)
    
    cgs_ortho_error = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(3), ord='fro')
    mgs_ortho_error = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(3), ord='fro')
    
    print(f"Classical GS Orthogonality Error ||Q^T Q - I||: {cgs_ortho_error:.4e}")
    print(f"Modified GS Orthogonality Error  ||Q^T Q - I||: {mgs_ortho_error:.4e}")
    assert mgs_ortho_error <= cgs_ortho_error
```

---

### Exercise 2: Orthogonal Projection onto Subspaces
**Concepts**: Projection matrix $P = A(A^T A)^{-1}A^T$, idempotence ($P^2 = P$), symmetry ($P^T = P$).

#### Problem Specification
1. Let subspace $\mathcal{S} = \text{span}\{a_1, a_2\} \subset \mathbb{R}^3$, where $a_1 = [1, 2, 1]^T$ and $a_2 = [0, 1, 1]^T$.
2. Derive and construct the projection matrix $P_\mathcal{S}$.
3. Verify computationally that $P_\mathcal{S}^2 = P_\mathcal{S}$ and $P_\mathcal{S}^T = P_\mathcal{S}$.
4. Decompose an arbitrary vector $y = [3, 1, 4]^T$ into orthogonal components: $y = \hat{y} + y_\perp$, where $\hat{y} \in \mathcal{S}$ and $y_\perp \in \mathcal{S}^\perp$. Verify that $\langle \hat{y}, y_\perp \rangle = 0$.

#### Verification Solution
```python
import numpy as np

def compute_projection_matrix(A: np.ndarray) -> np.ndarray:
    return A @ np.linalg.inv(A.T @ A) @ A.T

# Verification Run
if __name__ == "__main__":
    A = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    P = compute_projection_matrix(A)
    
    # Verify idempotence & symmetry
    assert np.allclose(P @ P, P), "Idempotence condition failed"
    assert np.allclose(P.T, P), "Symmetry condition failed"
    
    y = np.array([3.0, 1.0, 4.0])
    y_hat = P @ y
    y_perp = y - y_hat
    
    inner_product = np.dot(y_hat, y_perp)
    print("Projected y_hat:", y_hat)
    print("Residual orthogonal y_perp:", y_perp)
    print(f"Orthogonality check <y_hat, y_perp>: {inner_product:.4e}")
    assert np.isclose(inner_product, 0.0)
```

---

## Tier 2: SVD, Pseudo-Inverses & Spectral Theory

### Exercise 3: Full SVD via Eigendecomposition of Gram Matrices
**Concepts**: Singular values $\sigma_i = \sqrt{\lambda_i(A^T A)}$, left/right singular vectors $U$ and $V$.

#### Problem Specification
Given an arbitrary rectangular matrix $X \in \mathbb{R}^{m \times n}$:
1. Compute the eigenvalues and eigenvectors of $X^T X$ to construct right singular vectors $V$.
2. Compute the nonzero singular values $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_r)$.
3. Construct the left singular vectors $u_i = \frac{1}{\sigma_i} X v_i$.
4. Reconstruct the low-rank approximation $\hat{X}_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ and verify the Eckart-Young-Mirsky Theorem: $\|\hat{X}_k - X\|_F^2 = \sum_{j=k+1}^r \sigma_j^2$.

#### Verification Solution
```python
import numpy as np

def manual_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = X.shape
    # 1. Eigendecomposition of X^T X
    gram_right = X.T @ X
    eigenvalues, V = np.linalg.eigh(gram_right)
    
    # Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[sort_idx], 0.0)
    V = V[:, sort_idx]
    
    # Nonzero singular values
    singular_values = np.sqrt(eigenvalues)
    nonzero_mask = singular_values > 1e-10
    s_active = singular_values[nonzero_mask]
    V_active = V[:, nonzero_mask]
    
    # Compute U vectors: u_i = (1 / sigma_i) * X * v_i
    U_active = np.zeros((m, len(s_active)))
    for i in range(len(s_active)):
        U_active[:, i] = (X @ V_active[:, i]) / s_active[i]
        
    return U_active, s_active, V_active.T

# Verification Run
if __name__ == "__main__":
    np.random.seed(101)
    X = np.random.randn(5, 3)
    U, s, Vt = manual_svd(X)
    
    X_reconstructed = U @ np.diag(s) @ Vt
    rec_error = np.linalg.norm(X - X_reconstructed, ord='fro')
    print("Singular values:", s)
    print(f"Manual SVD Reconstruction Frobenius Error: {rec_error:.4e}")
    assert np.isclose(rec_error, 0.0)
```

---

### Exercise 4: Moore-Penrose Pseudo-Inverse & Minimum-Norm Least Squares
**Concepts**: Over-determined and under-determined systems, $A^+ = V \Sigma^+ U^T$, SVD pseudo-inverse.

#### Problem Specification
When a linear system $Ax = b$ is under-determined (infinitely many solutions), the Moore-Penrose pseudo-inverse $x^+ = A^+ b$ selects the unique solution with minimal Euclidean norm $\|x\|_2$:
1. Implement `svd_pinv(A, tol=1e-15)` from scratch using singular value truncation.
2. Given $A \in \mathbb{R}^{2 \times 4}$ and $b \in \mathbb{R}^2$, compute $x^+ = A^+ b$.
3. Perturb $x^+$ by arbitrary null-space vectors $v \in \text{null}(A)$ and prove that $\|x^+ + v\|_2 > \|x^+\|_2$ for all $v \neq 0$.

#### Verification Solution
```python
import numpy as np
from scipy.linalg import null_space

def svd_pinv(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.array([1.0 / val if val > tol else 0.0 for val in s])
    return Vt.T @ np.diag(s_inv) @ U.T

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    A = np.array([
        [1.0, 2.0, 1.0, 3.0],
        [2.0, 1.0, 0.0, 4.0]
    ])
    b = np.array([5.0, 7.0])
    
    A_pinv = svd_pinv(A)
    x_opt = A_pinv @ b
    
    # Verify Ax = b
    assert np.allclose(A @ x_opt, b), "System residual not zero"
    
    # Extract null-space basis
    ns = null_space(A)
    # Form arbitrary perturbation
    perturbation = ns @ np.array([0.5, -0.8])
    x_perturbed = x_opt + perturbation
    
    assert np.allclose(A @ x_perturbed, b), "Perturbed solution must still satisfy Ax=b"
    print(f"Norm of x_opt:       {np.linalg.norm(x_opt):.6f}")
    print(f"Norm of x_perturbed: {np.linalg.norm(x_perturbed):.6f}")
    assert np.linalg.norm(x_opt) < np.linalg.norm(x_perturbed)
```

---

## Tier 3: Matrix Calculus & Autodiff Engine Verification

### Exercise 5: Matrix Differential Verification of Linear Layer
**Concepts**: $\nabla_W \frac{1}{2}\|XW - Y\|_F^2 = X^T(XW - Y)$, Jacobian tensors, finite difference checks.

#### Problem Specification
1. Consider the linear regression objective function $f(W) = \frac{1}{2} \|XW - Y\|_F^2$ where $X \in \mathbb{R}^{N \times D}$, $W \in \mathbb{R}^{D \times K}$, and $Y \in \mathbb{R}^{N \times K}$.
2. Derive analytically: $\frac{\partial f}{\partial W} = X^T (XW - Y)$.
3. Implement `gradient_finite_difference(f, W, epsilon=1e-5)` to numerically approximate $\frac{\partial f}{\partial W_{i,j}}$.
4. Prove that the relative Frobenius norm error between the analytical gradient and the numerical gradient satisfies:
   $$\frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_F}{\|\nabla_{\text{analytical}}\|_F + \|\nabla_{\text{numerical}}\|_F} < 10^{-7}$$

#### Verification Solution
```python
import numpy as np

def analytical_gradient(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X.T @ (X @ W - Y)

def loss_fn(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    return 0.5 * float(np.sum((X @ W - Y) ** 2))

def numerical_gradient(X: np.ndarray, W: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad_num = np.zeros_like(W)
    D, K = W.shape
    for i in range(D):
        for j in range(K):
            W_pos = W.copy()
            W_neg = W.copy()
            W_pos[i, j] += eps
            W_neg[i, j] -= eps
            grad_num[i, j] = (loss_fn(X, W_pos, Y) - loss_fn(X, W_neg, Y)) / (2.0 * eps)
    return grad_num

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    N, D, K = 50, 8, 3
    X = np.random.randn(N, D)
    W = np.random.randn(D, K)
    Y = np.random.randn(N, K)
    
    grad_analytic = analytical_gradient(X, W, Y)
    grad_numeric = numerical_gradient(X, W, Y, eps=1e-6)
    
    rel_error = np.linalg.norm(grad_analytic - grad_numeric) / (
        np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)
    )
    print(f"Gradient Check Relative Error: {rel_error:.4e}")
    assert rel_error < 1e-7
```
