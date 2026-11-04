# Week 2: Graduate-Level Problem Set — Tree Algorithms, Split Criteria & Gradient Boosting Mathematics

Welcome to the Month 2 Week 2 Problem Set. This suite evaluates your mastery of non-linear tree-based machine learning: exact entropy and Gini impurity splits, second-order Taylor expansion of gradient boosted trees (XGBoost/LightGBM split formulas), optimal leaf weight derivation, and histogram-based binning.

---

## Tier 1: Core Mechanics & Tree Split Formulations

### Exercise 1: Shannon Entropy, Information Gain & Gini Impurity Derivations
**Concepts**: Shannon entropy $H(S) = -\sum p_k \log_2 p_k$, Gini impurity $G(S) = 1 - \sum p_k^2$, variance reduction for regression.

#### Problem Specification
Consider a binary classification node containing 100 samples with 60 positive class instances ($p_1 = 0.60$) and 40 negative class instances ($p_0 = 0.40$).
1. Compute the parent node's:
   - Gini Impurity: $G_{\text{parent}} = 1 - (0.60^2 + 0.40^2)$.
   - Shannon Entropy: $H_{\text{parent}} = -(0.60 \log_2 0.60 + 0.40 \log_2 0.40)$.
2. Suppose a candidate split on feature $X_1 \le 3.5$ partitions the samples into:
   - Left Child: 50 samples (45 positive, 5 negative).
   - Right Child: 50 samples (15 positive, 35 negative).
3. Compute the Information Gain $\text{IG}(S, X_1)$ and Gini Gain $\Delta G(S, X_1)$.
4. Implement a Python function `best_split_entropy(X, y)` that scans all unique continuous feature thresholds and returns the optimal threshold maximizing Information Gain.

#### Verification Solution
```python
import numpy as np

def entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def find_best_split(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    sorted_idx = np.argsort(x)
    x_sort, y_sort = x[sorted_idx], y[sorted_idx]
    
    parent_ent = entropy(y)
    n = len(y)
    best_ig = -1.0
    best_thresh = float(x_sort[0])
    
    for i in range(1, n):
        if x_sort[i] == x_sort[i - 1]:
            continue
        thresh = (x_sort[i] + x_sort[i - 1]) / 2.0
        y_left, y_right = y_sort[:i], y_sort[i:]
        
        ig = parent_ent - (len(y_left) / n * entropy(y_left) + len(y_right) / n * entropy(y_right))
        if ig > best_ig:
            best_ig = ig
            best_thresh = thresh
            
    return best_thresh, best_ig

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([0,   0,   0,   0,   1,   1,   1,   1])
    thresh, ig = find_best_split(x, y)
    print(f"Optimal Split Threshold: {thresh} | Max Information Gain: {ig:.4f}")
    assert thresh == 4.5
    assert np.isclose(ig, 1.0)
```

---

## Tier 2: 2nd-Order Taylor Expansion & Gradient Boosting

### Exercise 2: Mathematical Derivation of Optimal Leaf Weight & Split Gain
**Concepts**: 2nd-order Taylor approximation, pseudo-residuals $g_i = \frac{\partial \ell}{\partial \hat{y}_i}$, Hessian $h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^2}$, structural score $\mathcal{L}_{\text{split}}$.

#### Problem Specification
In modern Gradient Boosted Decision Trees (GBDT), the objective at tree step $t$ is approximated via a 2nd-order Taylor expansion:
$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{\lambda}{2} \sum_{j=1}^T w_j^2$$
1. Let $I_j = \{i \mid q(x_i) = j\}$ be the instance set assigned to leaf $j$. Prove that the optimal leaf weight $w_j^*$ is:
   $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} = -\frac{G_j}{H_j + \lambda}$$
2. Substitute $w_j^*$ back into the objective to prove that the optimal structural score is:
   $$\mathcal{L}^* = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$
3. Prove that for a candidate split into Left ($L$) and Right ($R$) subsets, the split gain $\text{Gain}$ is:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
4. Implement a vectorized split evaluation routine and verify that positive regularization $\lambda > 0$ strictly shrinks leaf weights toward zero.

#### Verification Solution
```python
import numpy as np

def compute_leaf_weight(g: np.ndarray, h: np.ndarray, reg_lambda: float) -> float:
    return float(-np.sum(g) / (np.sum(h) + reg_lambda))

def compute_split_gain(g_left: np.ndarray, h_left: np.ndarray, 
                       g_right: np.ndarray, h_right: np.ndarray, 
                       reg_lambda: float, gamma: float) -> float:
    G_L, H_L = float(np.sum(g_left)), float(np.sum(h_left))
    G_R, H_R = float(np.sum(g_right)), float(np.sum(h_right))
    G_total = G_L + G_R
    H_total = H_L + H_R
    
    score_left = (G_L ** 2) / (H_L + reg_lambda)
    score_right = (G_R ** 2) / (H_R + reg_lambda)
    score_parent = (G_total ** 2) / (H_total + reg_lambda)
    
    gain = 0.5 * (score_left + score_right - score_parent) - gamma
    return gain

# Verification Run
if __name__ == "__main__":
    # Test gradients for 10 samples
    g = np.array([-0.8, -0.6, -0.7, -0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    h = np.ones(10) * 0.25  # Standard for binary log-loss
    
    w_unreg = compute_leaf_weight(g[:4], h[:4], reg_lambda=0.0)
    w_reg = compute_leaf_weight(g[:4], h[:4], reg_lambda=2.0)
    
    print(f"Unregularized Leaf Weight: {w_unreg:.4f}")
    print(f"Regularized Leaf Weight:   {w_reg:.4f}")
    assert abs(w_reg) < abs(w_unreg), "L2 regularization must shrink leaf weight"
    
    gain = compute_split_gain(g[:4], h[:4], g[4:], h[4:], reg_lambda=1.0, gamma=0.1)
    print(f"Computed Split Gain: {gain:.4f}")
    assert gain > 0.0
```

---

## Tier 3: Histogram-Based Gradient Boosting Split Engine

### Exercise 3: Histogram Binning & Split Search (LightGBM Mechanics)
**Concepts**: Fast feature quantization into $K=256$ bins, histogram subtraction trick ($H_{\text{right}} = H_{\text{parent}} - H_{\text{left}}$).

#### Problem Specification
Standard exact greedy tree building requires sorting continuous values at every node ($O(n \log n)$). Modern engines quantize continuous features into integer histogram bins ($uint8$) and aggregate gradients:
1. Implement `quantize_feature(x, max_bins=256)`.
2. Construct histogram tables aggregating $G_{\text{bin}} = \sum_{i \in \text{bin}} g_i$ and $H_{\text{bin}} = \sum_{i \in \text{bin}} h_i$.
3. Demonstrate the **Histogram Subtraction Trick**: when splitting a node, compute the histogram for the smaller child directly, and obtain the histogram for the larger child in $O(K)$ time via:
   $$H_{\text{larger}} = H_{\text{parent}} - H_{\text{smaller}}$$

#### Verification Solution
```python
import numpy as np

def build_gradient_histogram(bin_indices: np.ndarray, g: np.ndarray, h: np.ndarray, num_bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    G_hist = np.zeros(num_bins, dtype=np.float64)
    H_hist = np.zeros(num_bins, dtype=np.float64)
    
    np.add.at(G_hist, bin_indices, g)
    np.add.at(H_hist, bin_indices, h)
    return G_hist, H_hist

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    n = 100_000
    num_bins = 16
    bins = np.random.randint(0, num_bins, size=n)
    g = np.random.randn(n)
    h = np.random.uniform(0.1, 1.0, size=n)
    
    # Parent histogram
    G_parent, H_parent = build_gradient_histogram(bins, g, h, num_bins)
    
    # Partition into left and right
    left_mask = bins < 6
    right_mask = ~left_mask
    
    G_left, H_left = build_gradient_histogram(bins[left_mask], g[left_mask], h[left_mask], num_bins)
    
    # Subtraction trick for right child
    G_right_sub = G_parent - G_left
    H_right_sub = H_parent - H_left
    
    # Exact calculation for right child
    G_right_exact, H_right_exact = build_gradient_histogram(bins[right_mask], g[right_mask], h[right_mask], num_bins)
    
    assert np.allclose(G_right_sub, G_right_exact)
    assert np.allclose(H_right_sub, H_right_exact)
    print("Histogram Subtraction Trick successfully verified with machine precision!")
```
