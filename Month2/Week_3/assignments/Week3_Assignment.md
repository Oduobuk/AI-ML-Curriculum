# Week 3 Production Engineering Assignment: Leak-Proof Purged CV & Bayesian Hyperparameter Optimization Suite

## Executive Overview & Architectural Motivation
The single largest reason machine learning models fail when transitioned from staging to live production is **evaluation leakage**. Shuffling temporal transactions, calculating global imputation statistics before splitting folds, and over-optimizing hyperparameter configs against un-purged test folds create devastating overconfidence.

In this assignment, you will architect an industrial-grade **Leak-Proof Validation & Bayesian Tuning Suite** for mission-critical tabular forecasting and fraud classification.

You will implement a custom `PurgedGroupTimeSeriesSplit` cross-validation engine with embargo buffers, build an **Optuna** multi-objective TPE optimization study, apply cost-sensitive class reweighting on extreme imbalance ($1:200$), and generate calibrated probability reliability diagrams.

---

## Technical Specifications & Architecture Diagram

```
                 Continuous Time-Series Transaction Matrix
                 [Sorted by Transaction Timestamp $t_0 \dots t_N$]
                                        │
                                        ▼
                 ┌───────────────────────────────────────────┐
                 │       PurgedGroupTimeSeriesSplit Engine   │
                 │   - Expanding / Rolling Train Window      │
                 │   - Purge Gap: Drops overlapping labels   │
                 │   - Embargo Buffer: Prevents serial leak  │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Optuna Bayesian Study (TPE)         │
                 │   - Hyperparameters: max_depth, lr, reg   │
                 │   - Objective: Maximize PR-AUC (Average P)│
                 │   - Pruning: MedianPruner for slow trials │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │      Cost-Sensitive Imbalance Calibrator  │
                 │   - Optimal Decision Threshold Finding    │
                 │   - Platt Scaling / Isotonic Calibration  │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 Reliability Diagrams & Production Model Artifacts
```

---

## Detailed Deliverable Requirements

### 1. Purged Group TimeSeries Cross-Validation Engine (`splitters.py`)
- Implement `PurgedGroupTimeSeriesSplit`:
  - Partitions temporal datasets across $K$ sequential test folds.
  - Implements **Purging**: removes observations from the training set whose label calculation windows overlap with the test set evaluation window.
  - Implements **Embargo**: drops observations immediately following the test set to account for long-memory autoregressive correlation.
- Enforce strict monotonic timestamp ordering assertions.

### 2. Optuna Bayesian Hyperparameter Study (`tuning.py`)
- Configure an Optuna study with `TPESampler(seed=42)` and `MedianPruner()`.
- Search space:
  - `max_depth` $\in [3, 10]$
  - `learning_rate` $\in [10^{-3}, 0.3]$ (log-scale)
  - `colsample_bytree` $\in [0.4, 1.0]$
  - `reg_lambda` $\in [10^{-2}, 10.0]$
- Optimize strictly for out-of-fold **Average Precision (PR-AUC)** rather than ROC-AUC or Accuracy.

### 3. Probability Calibration & Decision Boundary Audit (`calibration.py`)
- Raw tree model outputs frequently suffer from uncalibrated probabilities near extremes.
- Fit `CalibratedClassifierCV` (isotonic regression / Platt scaling) on purged validation predictions.
- Calculate and plot:
  - Reliability curve (observed empirical frequency vs mean predicted probability across 10 bins).
  - Brier Score loss: $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
  - Precision-Recall F1 optimal decision threshold cutoff.

---

## Starter Code Template & Submission Skeleton

```python
"""
purged_cv_optuna_tuning.py - Production Validation & Tuning Engine
"""
import numpy as np
import optuna
from typing import Generator, Tuple, List
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

class PurgedGroupTimeSeriesSplit:
    """
    Purged & Embargoed Time-Series Cross Validation Splitter.
    """
    def __init__(self, n_splits: int = 4, purge_samples: int = 50, embargo_samples: int = 25):
        self.n_splits = n_splits
        self.purge_samples = purge_samples
        self.embargo_samples = embargo_samples

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for k in range(1, self.n_splits + 1):
            # Test split indices
            test_start = k * fold_size
            test_end = (k + 1) * fold_size
            test_idx = np.arange(test_start, test_end)
            
            # Train split: everything before test_start - purge_samples
            train_end = max(0, test_start - self.purge_samples)
            train_idx = np.arange(0, train_end)
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx

def run_optuna_hpo(X: np.ndarray, y: np.ndarray, n_trials: int = 15) -> optuna.Study:
    cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_samples=20, embargo_samples=10)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 5.0, log=True),
            "random_state": 42
        }
        
        oof_scores = []
        for train_idx, test_idx in cv.split(X):
            clf = HistGradientBoostingClassifier(**params)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict_proba(X[test_idx])[:, 1]
            score = average_precision_score(y[test_idx], preds)
            oof_scores.append(score)
            
        return float(np.mean(oof_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study

# Verification Run
if __name__ == "__main__":
    np.random.seed(42)
    # Generate imbalanced temporal sequence
    N = 10_000
    X = np.random.randn(N, 8)
    # True relationship with noise
    prob = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 1.2 * X[:, 1] - 3.5)))
    y = (np.random.rand(N) < prob).astype(int)
    
    print(f"Dataset generated: {N} records | Positive prevalence: {np.mean(y)*100:.2f}%")
    
    study = run_optuna_hpo(X, y, n_trials=5)
    print("Best Trial PR-AUC:", study.best_value)
    print("Best Parameters:", study.best_params)
    assert study.best_value > 0.15
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Purged Time-Series Splitter** | 35 | Strict zero-leakage temporal ordering, purge gap, and embargo buffer implementation. |
| **Optuna Bayesian Study Design** | 25 | Proper parameter search space, TPE sampler, PR-AUC objective, reproducible seeds. |
| **Handling Extreme Class Imbalance** | 20 | Focus on Precision-Recall geometry, cost matrix re-weighting, and threshold optimization. |
| **Probability Calibration (Brier Score)** | 10 | Proper reliability curve generation and calibration error verification. |
| **Verification & Test Coverage** | 10 | Fully automated end-to-end verification script with assertions. |
