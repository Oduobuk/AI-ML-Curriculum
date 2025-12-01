# Month 3, Week 3 Cheatsheet: Model Validation & Hyperparameter Tuning

## 1. Bias-Variance Trade-off

*   **Bias:** Error from approximating a real-world problem with a simplified model.
    *   **High Bias:** Model is too simple (underfitting), performs poorly on train and test data.
*   **Variance:** Error from model's sensitivity to specific training data.
    *   **High Variance:** Model is too complex (overfitting), performs well on train but poorly on test data.
*   **Goal:** Find a balance to minimize total generalization error.

---

## 2. Model Validation

*   **Purpose:** Assess how well a model generalizes to unseen data.

### a) Train-Test Split

*   **Process:** Divide data into training set (for model learning) and test set (for unbiased evaluation).
*   **Limitation:** Performance estimate can be sensitive to the specific split.

### b) Cross-Validation (CV)

*   **Purpose:** Provides a more robust and reliable estimate of model performance.
*   **K-Fold Cross-Validation:**
    1.  Data split into `K` equal-sized folds.
    2.  Model trained `K` times; each time, one fold is validation, `K-1` folds are training.
    3.  Performance averaged across `K` iterations.
*   **Stratified K-Fold CV:** Ensures each fold has the same proportion of class labels as the original dataset (good for imbalanced data).
*   **Advantages:** More stable performance estimate, uses all data for both training and validation.

---

## 3. Hyperparameter Tuning

*   **Hyperparameters:** Parameters set *before* training (e.g., learning rate, `n_estimators`, `max_depth`).
*   **Goal:** Find the optimal combination of hyperparameters for best model performance.

### a) Grid Search

*   **Process:** Defines a grid of hyperparameter values. Evaluates every possible combination.
*   **Advantages:** Exhaustive, guarantees finding the best combination within the grid.
*   **Disadvantages:** Computationally expensive, scales poorly with many hyperparameters.

### b) Random Search

*   **Process:** Defines a range/distribution for each hyperparameter. Samples a fixed number of random combinations.
*   **Advantages:** Often finds good hyperparameters faster than Grid Search, especially when some hyperparameters are more influential.
*   **Disadvantages:** Not exhaustive, might miss optimal combinations.

### c) Other Methods

*   **Bayesian Optimization:** Uses a probabilistic model to intelligently explore the hyperparameter space.
*   **Genetic Algorithms:** Uses evolutionary principles to search for optimal hyperparameters.

---

## 4. Overfitting & Underfitting

*   **Overfitting:** Model learns training data (including noise) too well, performs poorly on new data (high variance, low bias).
*   **Underfitting:** Model is too simple, fails to capture underlying patterns, performs poorly on both training and test data (high bias, high variance).
*   **Generalization:** The ability of a model to perform well on unseen data.

---

## 5. MLflow: ML Lifecycle Management

*   **Purpose:** Open-source platform to manage the end-to-end machine learning lifecycle.

### a) MLflow Tracking

*   **Functionality:** Records and queries experiments (parameters, metrics, code versions, artifacts).
*   **Benefit:** Reproducibility and comparison of runs.

### b) MLflow Projects

*   **Functionality:** Packages ML code in a reusable and reproducible format.

### c) MLflow Models

*   **Functionality:** Manages and deploys ML models from various libraries to diverse deployment tools.

### d) MLflow Model Registry

*   **Functionality:** Centralized store for collaborative model management (versioning, stage transitions).
