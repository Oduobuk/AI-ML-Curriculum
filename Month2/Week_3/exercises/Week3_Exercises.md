# Week 3: Graduate-Level Problem Set — Cross-Validation, Class Imbalance & Optuna Hyperparameter Optimization

Welcome to the Month 2 Week 3 Problem Set. This suite rigorous covers production model validation and hyperparameter calibration: time-series data leakage prevention, purged/embargoed cross-validation, cost-sensitive classification vs SMOTE, precision-recall curve geometry, and Bayesian Tree-structured Parzen Estimator (TPE) optimization.

---

## Tier 1: Core Mechanics & Validation Leakage

### Exercise 1: Time-Series Leakage & Purged Cross-Validation
**Concepts**: Autoregressive leakage, lookahead bias, PurgedGroupTimeSeriesSplit, embargo windows.

#### Problem Specification
Standard $K$-Fold cross-validation randomly shuffles data, allowing future records to leak into past training folds:
1. Generate an autoregressive time-series process $y_t = 0.85 y_{t-1} + \epsilon_t$.
2. Train a lag-1 linear model using:
   - Naive Shuffled $K$-Fold CV ($k=5$).
   - Temporal Blocked Walk-Forward CV ($k=5$).
3. Demonstrate how random shuffling produces artificially deflated test Mean Squared Error (MSE) due to temporal autocorrelation leakage.
4. Implement `PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=10, embargo_window=5)` in pure Python.

#### Verification Solution
```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def demonstrate_temporal_leakage():
    np.random.seed(42)
    n = 1000
    # Autoregressive AR(1) process
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.85 * y[t - 1] + np.random.normal(0, 0.5)
        
    X = y[:-1].reshape(-1, 1)
    y_target = y[1:]
    
    # 1. Leaky Shuffled K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    leaky_mse = []
    for train_idx, test_idx in kf.split(X):
        model = Ridge().fit(X[train_idx], y_target[train_idx])
        preds = model.predict(X[test_idx])
        leaky_mse.append(np.mean((y_target[test_idx] - preds) ** 2))
        
    # 2. Strict Walk-Forward TimeSplit
    walk_forward_mse = []
    fold_size = len(X) // 5
    for k in range(1, 5):
        train_idx = np.arange(0, k * fold_size)
        test_idx = np.arange(k * fold_size, (k + 1) * fold_size)
        model = Ridge().fit(X[train_idx], y_target[train_idx])
        preds = model.predict(X[test_idx])
        walk_forward_mse.append(np.mean((y_target[test_idx] - preds) ** 2))
        
    print(f"Leaky Shuffled CV MSE:      {np.mean(leaky_mse):.4f}")
    print(f"Strict Walk-Forward CV MSE: {np.mean(walk_forward_mse):.4f}")
    assert np.mean(leaky_mse) <= np.mean(walk_forward_mse)

if __name__ == "__main__":
    demonstrate_temporal_leakage()
```

---

### Exercise 2: Extreme Class Imbalance & Precision-Recall Curve Geometry
**Concepts**: ROC-AUC distortion under severe imbalance, Precision-Recall AUC (PR-AUC), Brier Score calibration.

#### Problem Specification
In fraud detection ($1:1000$ prevalence), a naive classifier predicting all zeros achieves $99.9\%$ accuracy and an ROC-AUC of $\sim 0.50$ to $0.90$ despite providing zero utility:
1. Generate an imbalanced dataset with $N = 50,000$ and $0.2\%$ positive cases (100 positives, 49,900 negatives).
2. Train a baseline logistic regression model.
3. Compute and compare:
   - ROC-AUC: $\int \text{TPR} \, d(\text{FPR})$.
   - Average Precision (PR-AUC): $\int \text{Precision} \, d(\text{Recall})$.
4. Explain why PR-AUC penalizes false positives much more severely than ROC-AUC in the presence of overwhelming negatives.

#### Verification Solution
```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def evaluate_imbalance_metrics():
    np.random.seed(42)
    n_neg = 49_900
    n_pos = 100
    
    # Ground truth
    y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    
    # Model predictions with high false positive rate
    preds_neg = np.random.beta(0.5, 5.0, size=n_neg)
    preds_pos = np.random.beta(2.0, 2.0, size=n_pos)
    y_scores = np.concatenate([preds_neg, preds_pos])
    
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    print(f"ROC-AUC: {roc_auc:.4f} (misleadingly optimistic)")
    print(f"PR-AUC:  {pr_auc:.4f} (realistically revealing severe false positives)")
    assert roc_auc > 0.80
    assert pr_auc < 0.20

if __name__ == "__main__":
    evaluate_imbalance_metrics()
```

---

## Tier 2: Bayesian Optimization & TPE Mechanics

### Exercise 3: Tree-Structured Parzen Estimator (TPE) Simulation
**Concepts**: Nonparametric density estimation, Gaussian mixture models, expected improvement $\text{EI}(x) = \frac{\ell(x)}{g(x)}$.

#### Problem Specification
Bayesian optimization models hyperparameter evaluations by splitting historical trials into good ($\ell(x)$) and bad ($g(x)$) distributions based on quantile $\gamma = 0.15$:
1. Write a function `tpe_sample(history, gamma=0.20)` that:
   - Partitions evaluated hyperparameters $\theta_i$ into Top $20\%$ ($\ell$) and Remaining $80\%$ ($g$).
   - Fits 1D Gaussian Kernel Density Estimators (KDE) to both sets.
   - Generates candidate samples from $\ell(\theta)$ and selects the candidate maximizing the acquisition ratio $\frac{\ell(\theta)}{g(\theta)}$.
2. Optimize the 1D function $f(x) = (x - 3)^2 + \sin(5x)$ across 25 iterations.

#### Verification Solution
```python
import numpy as np
from scipy.stats import gaussian_kde

def tpe_optimize_1d(objective_fn, bounds=(-5.0, 5.0), n_init=10, n_iter=25, gamma=0.25):
    np.random.seed(42)
    # 1. Random initial trials
    X_history = list(np.random.uniform(bounds[0], bounds[1], size=n_init))
    y_history = [objective_fn(x) for x in X_history]
    
    for it in range(n_iter):
        split_idx = int(len(y_history) * gamma)
        sorted_indices = np.argsort(y_history)
        
        good_pts = np.array(X_history)[sorted_indices[:max(split_idx, 2)]]
        bad_pts = np.array(X_history)[sorted_indices[max(split_idx, 2):]]
        
        # Fit KDEs with smoothing
        kde_good = gaussian_kde(good_pts, bw_method=0.5)
        kde_bad = gaussian_kde(bad_pts, bw_method=0.5)
        
        # Sample candidate points from good KDE
        candidates = np.clip(kde_good.resample(100)[0], bounds[0], bounds[1])
        # Acquire point maximizing ratio l(x) / g(x)
        ratio = kde_good(candidates) / (kde_bad(candidates) + 1e-6)
        best_cand = float(candidates[np.argmax(ratio)])
        
        # Evaluate
        y_cand = objective_fn(best_cand)
        X_history.append(best_cand)
        y_history.append(y_cand)
        
    best_overall_x = X_history[np.argmin(y_history)]
    best_overall_y = np.min(y_history)
    return best_overall_x, best_overall_y

# Verification Run
if __name__ == "__main__":
    def test_obj(x):
        return float((x - 2.0)**2 + 0.5 * np.sin(4.0 * x))
        
    x_opt, y_opt = tpe_optimize_1d(test_obj)
    print(f"Optimal x found by TPE: {x_opt:.4f} | Optimal objective: {y_opt:.4f}")
    assert abs(x_opt - 2.0) < 0.5
```
