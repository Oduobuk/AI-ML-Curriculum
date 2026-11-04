# Week 3 Production Engineering Assignment: Low-Rank Matrix Factorization & Latent Factor Recommendation Engine

## Executive Overview & Architectural Motivation
Modern enterprise recommendation architectures (e.g., Netflix Prize models, Spotify collaborative filtering, e-commerce item retrieval) operate on massive user-item interaction matrices $R \in \mathbb{R}^{m \times n}$ where $99\%+$ of entries are unobserved (sparse).

In this assignment, you will engineer a complete, from-scratch **Low-Rank Matrix Factorization Engine** utilizing truncated Singular Value Decomposition (SVD) and Alternating Least Squares (ALS) with $L_2$ Tikhonov regularization.

You will demonstrate the mathematical equivalence between low-rank matrix projection, dimensionality reduction (Eckart-Young-Mirsky Theorem), and latent feature space alignment.

---

## Technical Specifications & Mathematical Formulation

Given an observed sparse rating matrix $R \in \mathbb{R}^{m \times n}$, we seek to factorize $R$ into a user latent matrix $U \in \mathbb{R}^{m \times k}$ and an item latent matrix $V \in \mathbb{R}^{n \times k}$ (with latent rank $k \ll \min(m, n)$) by minimizing the regularized squared loss over observed indices $\Omega$:

$$\mathcal{L}(U, V) = \sum_{(u, i) \in \Omega} \left(R_{u,i} - u_u^T v_i\right)^2 + \lambda \left(\sum_{u=1}^m \|u_u\|_2^2 + \sum_{i=1}^n \|v_i\|_2^2\right)$$

### Closed-Form Alternating Least Squares (ALS) Updates:
1. **Fix $V$, update each user vector $u_u$**:
   $$u_u = \left(\sum_{i \in \Omega_u} v_i v_i^T + \lambda I_k\right)^{-1} \sum_{i \in \Omega_u} R_{u, i} v_i$$
2. **Fix $U$, update each item vector $v_i$**:
   $$v_i = \left(\sum_{u \in \Omega_i} u_u u_u^T + \lambda I_k\right)^{-1} \sum_{u \in \Omega_i} R_{u, i} u_u$$

---

## Detailed Deliverable Requirements

### 1. Truncated SVD Baseline on Dense Mean-Imputed Data (`svd_recommender.py`)
- Implement randomized/truncated SVD to compute optimal rank-$k$ approximation $\hat{R} = U_k \Sigma_k V_k^T$.
- Compute the explained variance ratio:
  $$\text{EVR}_k = \frac{\sum_{j=1}^k \sigma_j^2}{\sum_{j=1}^{\min(m,n)} \sigma_j^2}$$
- Verify the Eckart-Young reconstruction bound against full SVD.

### 2. Sparse Alternating Least Squares (ALS) Engine (`als_engine.py`)
- Implement the closed-form ALS optimization algorithm in pure NumPy.
- Implement explicit regularization $\lambda$ to prevent latent vector explosion.
- Track train and validation Root Mean Square Error (RMSE) across iterations:
  $$\text{RMSE} = \sqrt{\frac{1}{|\Omega_{\text{test}}|} \sum_{(u, i) \in \Omega_{\text{test}}} (R_{u, i} - \hat{R}_{u, i})^2}$$

### 3. Latent Vector Similarity & Top-$N$ Retrieval (`retrieval.py`)
- Implement cosine similarity search across item latent vectors $v_i$:
  $$\text{sim}(v_i, v_j) = \frac{v_i^T v_j}{\|v_i\|_2 \|v_j\|_2}$$
- Given a user profile, compute predicted ratings across all unrated items and return the Top-$N$ items.

---

## Starter Code Template & Submission Skeleton

```python
"""
matrix_factorization_als.py - Production ALS Recommender
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class ALSConfig:
    latent_dim: int = 16
    reg_lambda: float = 0.05
    max_epochs: int = 15
    seed: int = 42

class ExplicitALSRecommender:
    def __init__(self, config: ALSConfig):
        self.config = config
        self.U: np.ndarray = np.empty(0)
        self.V: np.ndarray = np.empty(0)
        self.train_rmse_history: List[float] = []

    def fit(self, R: np.ndarray, mask: np.ndarray) -> "ExplicitALSRecommender":
        """
        Fits low-rank user/item matrices on observed ratings.
        R: [m x n] rating matrix
        mask: [m x n] boolean array where True indicates observed rating.
        """
        np.random.seed(self.config.seed)
        m, n = R.shape
        k = self.config.latent_dim
        lam = self.config.reg_lambda
        
        # Initialize latent matrices
        self.U = np.random.normal(scale=1.0 / np.sqrt(k), size=(m, k))
        self.V = np.random.normal(scale=1.0 / np.sqrt(k), size=(n, k))
        
        I_k = np.eye(k)

        for epoch in range(self.config.max_epochs):
            # 1. Update Users U
            for u in range(m):
                observed_items = np.where(mask[u, :])[0]
                if len(observed_items) == 0:
                    continue
                V_sub = self.V[observed_items, :]  # [len x k]
                R_sub = R[u, observed_items]       # [len]
                A_u = V_sub.T @ V_sub + lam * I_k  # [k x k]
                b_u = V_sub.T @ R_sub              # [k]
                self.U[u, :] = np.linalg.solve(A_u, b_u)

            # 2. Update Items V
            for i in range(n):
                observed_users = np.where(mask[:, i])[0]
                if len(observed_users) == 0:
                    continue
                U_sub = self.U[observed_users, :]  # [len x k]
                R_sub = R[observed_users, i]       # [len]
                A_i = U_sub.T @ U_sub + lam * I_k  # [k x k]
                b_i = U_sub.T @ R_sub              # [k]
                self.V[i, :] = np.linalg.solve(A_i, b_i)

            # Calculate RMSE over observed entries
            R_pred = self.U @ self.V.T
            diff = (R[mask] - R_pred[mask])
            rmse = np.sqrt(np.mean(diff ** 2))
            self.train_rmse_history.append(rmse)
            # print(f"Epoch {epoch + 1:02d} | Train RMSE: {rmse:.4f}")

        return self

    def predict(self) -> np.ndarray:
        return self.U @ self.V.T

    def recommend_top_n(self, user_idx: int, mask: np.ndarray, top_n: int = 5) -> List[Tuple[int, float]]:
        unseen_items = np.where(~mask[user_idx, :])[0]
        scores = self.U[user_idx, :] @ self.V[unseen_items, :].T
        sorted_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(unseen_items[idx]), float(scores[idx])) for idx in sorted_indices]

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic user-item matrix (100 users, 50 items)
    m, n = 100, 50
    true_U = np.random.randn(m, 5)
    true_V = np.random.randn(n, 5)
    full_R = np.clip(true_U @ true_V.T + 3.0, 1.0, 5.0)
    
    # 80% sparsity
    mask = np.random.rand(m, n) > 0.80
    observed_R = full_R * mask

    model = ExplicitALSRecommender(ALSConfig(latent_dim=5, reg_lambda=0.1, max_epochs=10))
    model.fit(observed_R, mask)
    
    print("Initial RMSE:", model.train_rmse_history[0])
    print("Final RMSE:  ", model.train_rmse_history[-1])
    assert model.train_rmse_history[-1] < model.train_rmse_history[0]
    
    recs = model.recommend_top_n(user_idx=0, mask=mask, top_n=3)
    print("Top 3 recommendations for User 0:", recs)
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Mathematical Derivation & Updates** | 30 | Correct normal equations derivation and ALS matrix solve ($A x = b$) implementation. |
| **Convergence & Regularization** | 25 | Monotonic convergence of training loss, stable inversion via ridge parameter $\lambda I_k$. |
| **SVD & Truncation Comparison** | 20 | Complete comparison of full vs truncated SVD explained variance. |
| **Top-N Inference & Cosine Retrieval** | 15 | Accurate dot-product scoring and filtering of previously rated items. |
| **Verification & Numerical Stability** | 10 | Comprehensive unit test suite on sparse synthetic matrices. |
