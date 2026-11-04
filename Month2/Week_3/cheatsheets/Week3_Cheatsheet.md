# Month 2, Week 3 Cheatsheet: Validation Strategies, Extreme Imbalance & Bayesian HPO (Optuna)

---

## 1. Cross-Validation Leakage Prevention

```
                     Purged & Embargoed Time-Series Cross Validation
  0 ────────────────────────────────────────────────────────────────────────► Time
 ┌───────────────────────┬────────────┬─────────────┬───────────┬─────────────┐
 │    Training Fold      │ Purge Gap  │  Test Fold  │  Embargo  │ Future Fold │
 └───────────────────────┴────────────┴─────────────┴───────────┴─────────────┘
 (Purge drops overlapping event labels; Embargo prevents autoregressive memory leak)
```

### Critical Leakage Modes
- **Lookahead Leakage**: Standard random $K$-Fold shuffles temporal data, letting future events predict past events.
- **Preprocessor Leakage**: Calling `scaler.fit(X)` or `target_encoder.fit(X, y)` before train/test splitting. Always fit transformers strictly on training folds.

---

## 2. Extreme Class Imbalance & Evaluation Geometry

In high-imbalance problems (prevalence $< 1\%$), Accuracy and ROC-AUC become dangerously optimistic:

### Precision-Recall Curve (PR-AUC / Average Precision)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **PR-AUC** isolates the minority class; false positives directly drive Precision down toward zero.
- **Brier Score** measures probability calibration:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### Cost-Sensitive Objective Reweighting
Assign cost matrix weights $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$ in binary cross-entropy:
$$\mathcal{L} = -\sum_{i=1}^N \left[ w_{\text{pos}} y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$

---

## 3. Bayesian Optimization via Tree-Structured Parzen Estimator (TPE)

Bayesian HPO models hyperparameter distributions using historical trial performances:
$$\text{Expected Improvement: } \text{EI}(x) = \frac{\ell(x)}{g(x)}$$
where:
- $\ell(x)$: Density of hyperparameters that yielded top $\gamma$ percentile performances (e.g. top $15\%$).
- $g(x)$: Density of hyperparameters that yielded remaining non-top performances.

---

## 4. Production Optuna Tuning Skeleton with Pruning

```python
import optuna

def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0)
    }
    
    # Evaluate across purged CV folds...
    for step, (tr_idx, val_idx) in enumerate(cv.split(X)):
        model.fit(X[tr_idx], y[tr_idx])
        score = evaluate_pr_auc(model, X[val_idx], y[val_idx])
        
        # Report for intermediate trial pruning
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return mean_score

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)
study.optimize(objective, n_trials=50)
```
