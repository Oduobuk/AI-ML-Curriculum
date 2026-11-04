# Month 2, Week 2 Cheatsheet: Decision Trees, Split Mechanics & Second-Order Gradient Boosting (GBDT)

---

## 1. Classical Impurity Criteria

For node $S$ with class proportions $p_k \in [0, 1]$ across $K$ classes:

### Shannon Entropy & Information Gain
$$H(S) = -\sum_{k=1}^K p_k \log_2(p_k), \qquad \text{IG}(S, A) = H(S) - \sum_{v \in \text{Children}} \frac{|S_v|}{|S|} H(S_v)$$

### Gini Impurity
$$G(S) = 1 - \sum_{k=1}^K p_k^2 = \sum_{j \neq k} p_j p_k$$

### Variance Reduction (Regression)
$$\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

---

## 2. Second-Order Taylor Expansion of Gradient Boosted Trees

Modern GBDTs (XGBoost, LightGBM, CatBoost) approximate arbitrary differentiable loss functions via a second-order Taylor expansion at boosting round $t$:
$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{\lambda}{2} \sum_{j=1}^T w_j^2$$
where:
$$g_i = \left[\frac{\partial \ell(y_i, \hat{y})}{\partial \hat{y}}\right]_{\hat{y}^{(t-1)}}, \qquad h_i = \left[\frac{\partial^2 \ell(y_i, \hat{y})}{\partial \hat{y}^2}\right]_{\hat{y}^{(t-1)}}$$

### Standard Loss Derivatives
| Objective | Loss $\ell(y, \hat{y})$ | 1st Order Gradient $g_i$ | 2nd Order Hessian $h_i$ |
|:---|:---|:---|:---|
| **Squared Error** | $\frac{1}{2}(y - \hat{y})^2$ | $\hat{y} - y$ | $1.0$ |
| **Binary Log-Loss** | $-y \ln p - (1-y)\ln(1-p)$ | $p - y$ where $p = \sigma(\hat{y})$ | $p(1 - p)$ |

---

## 3. Optimal Regularized Leaf Weight & Split Gain Formula

For instance set $I_j$ in leaf $j$, let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$.

### Optimal Leaf Weight $w_j^*$
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} = -\frac{G_j}{H_j + \lambda}$$

### Split Gain Metric
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
- Split is accepted if $\text{Gain} > 0$.
- Minimum child weight constraint: requires $H_L \ge \text{min\_child\_weight}$ and $H_R \ge \text{min\_child\_weight}$.

---

## 4. LightGBM Histogram Subtraction Optimization

Instead of sorting continuous features ($O(N \log N)$), features are binned into $K=256$ histogram buckets.
$$\text{Histogram for Larger Child: } H_{\text{larger}} = H_{\text{parent}} - H_{\text{smaller}}$$
- Reduces histogram construction complexity by $50\%$ at every internal tree split.
- Bins consume single bytes (`uint8`), enabling $8\times$ cache memory density.

---

## 5. Shrinkage Learning Rate & Base Score Initialization

```python
# Initial raw logit prediction for binary classification
base_score = np.log(p_mean / (1.0 - p_mean))

# Additive ensemble step with shrinkage factor eta
raw_preds = raw_preds + eta * tree.predict(X)
```
