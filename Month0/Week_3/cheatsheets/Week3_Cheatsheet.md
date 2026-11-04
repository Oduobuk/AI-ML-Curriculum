# Month 0, Week 3 Cheatsheet: Linear Algebra & Matrix Calculus for Machine Learning

---

## 1. Vector Spaces, Norms & Inner Products

### Norm Formulas in $\mathbb{R}^d$
$$\|x\|_1 = \sum_{i=1}^d |x_i|, \quad \|x\|_2 = \sqrt{\sum_{i=1}^d x_i^2}, \quad \|x\|_\infty = \max_i |x_i|, \quad \|x\|_p = \left(\sum_{i=1}^d |x_i|^p\right)^{1/p}$$

### Matrix Frobenius Norm
$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{i, j}^2} = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum_{i=1}^{\min(m, n)} \sigma_i^2}$$

### Cauchy-Schwarz & Triangle Inequalities
$$|\langle u, v \rangle| \le \|u\|_2 \|v\|_2, \qquad \|u + v\|_2 \le \|u\|_2 + \|v\|_2$$

---

## 2. Orthogonal Projections & QR Factorization

### Projection Operator onto Subspace $\mathcal{S} = \text{span}(A)$
$$P = A(A^T A)^{-1} A^T$$
- **Idempotent**: $P^2 = P$
- **Symmetric**: $P^T = P$
- **Orthogonal Complement**: $P_\perp = I - P$

### Gram-Schmidt Orthogonalization ($A = QR$)
```python
import numpy as np

def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    V = A.copy().astype(np.float64)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] -= R[i, j] * Q[:, i]
    return Q, R
```

---

## 3. Singular Value Decomposition (SVD) & Spectral Geometry

Every real matrix $X \in \mathbb{R}^{m \times n}$ factorizes into:
$$X = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T$$
where:
- $U \in \mathbb{R}^{m \times m}$: Orthonormal left singular vectors (eigenvectors of $X X^T$).
- $\Sigma \in \mathbb{R}^{m \times n}$: Diagonal singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.
- $V \in \mathbb{R}^{n \times n}$: Orthonormal right singular vectors (eigenvectors of $X^T X$).

### Eckart-Young-Mirsky Low-Rank Approximation Theorem
The optimal rank-$k$ approximation ($k < r$) minimizing Frobenius error is:
$$\hat{X}_k = \sum_{i=1}^k \sigma_i u_i v_i^T, \qquad \min_{\text{rank}(B)=k} \|X - B\|_F^2 = \|X - \hat{X}_k\|_F^2 = \sum_{j=k+1}^r \sigma_j^2$$

---

## 4. Moore-Penrose Pseudo-Inverse ($A^+$)

For any matrix $A = U \Sigma V^T$, its pseudo-inverse is:
$$A^+ = V \Sigma^+ U^T \quad \text{where } \Sigma_{i, i}^+ = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > \epsilon \\ 0 & \text{otherwise} \end{cases}$$

### Applications to Linear Systems $Ax = b$
1. **Over-determined ($m > n$, full column rank)**: $A^+ = (A^T A)^{-1} A^T$ (Least-Squares Solution).
2. **Under-determined ($m < n$, full row rank)**: $A^+ = A^T (A A^T)^{-1}$ (Minimum-Norm Solution).

---

## 5. Matrix Differential Calculus Identities for Deep Learning

| Scalar Objective Function $f(W)$ | Gradient $\nabla_W f(W)$ |
|:---|:---|
| $f(W) = a^T W b$ | $a b^T$ |
| $f(W) = x^T W x$ (symmetric $W$) | $2 x x^T$ |
| $f(W) = \frac{1}{2} \|X W - Y\|_F^2$ | $X^T (X W - Y)$ |
| $f(W) = \text{Tr}(W)$ | $I$ |
| $f(W) = \ln |W|$ (invertible $W$) | $W^{-T}$ |
| $f(W) = \frac{1}{2} \|W\|_F^2$ | $W$ |

### Vector-Jacobian Product (VJP) in Neural Layers
For a forward layer $y = W x$:
$$\nabla_x \mathcal{L} = W^T (\nabla_y \mathcal{L}), \qquad \nabla_W \mathcal{L} = (\nabla_y \mathcal{L}) x^T, \qquad \nabla_b \mathcal{L} = \nabla_y \mathcal{L}$$
