# Month 2, Week 4 Cheatsheet: Boosting & Support Vector Machines (SVM)

## 1. Boosting

*   **Purpose:** Ensemble technique that combines multiple "weak learners" sequentially to form a single "strong learner." Focuses on correcting errors of previous models.
*   **Sequential Training:** Models are trained one after another.
*   **Bias Reduction:** Primarily aims to reduce bias and convert weak learners into strong ones.

### a) AdaBoost (Adaptive Boosting)

*   **Weak Learner:** Often uses decision stumps (decision trees with `max_depth=1`).
*   **Mechanism:**
    1.  Starts with equal weights for all training samples.
    2.  Trains a weak learner.
    3.  Increases weights of misclassified samples, decreases weights of correctly classified samples.
    4.  Combines weak learners with weights based on their accuracy.
*   **Focus:** Iteratively improves performance on hard-to-classify samples.

### b) Gradient Boosting

*   **Mechanism:** Builds new models that predict the **residuals (errors)** of prior models. Each new model is trained to minimize the loss function by moving in the direction of the negative gradient.
*   **Generalization:** Highly flexible, can use various loss functions and weak learners.

### c) Advanced Gradient Boosting Implementations

*   **XGBoost (Extreme Gradient Boosting):**
    *   Optimized, scalable, and efficient.
    *   Includes regularization (L1/L2), parallel processing, tree pruning, and missing value handling.
    *   Known for high performance and speed.
*   **LightGBM:**
    *   Uses tree-based learning algorithms.
    *   Designed for speed and efficiency on large datasets (e.g., Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB)).
*   **CatBoost:**
    *   Excels at handling categorical features automatically.
    *   Uses ordered boosting to combat prediction shift.

---

## 2. Support Vector Machines (SVM)

*   **Purpose:** Powerful supervised learning model for **classification**, regression, and outlier detection.
*   **Key Idea:** Finds the optimal hyperplane that maximizes the margin between classes.

### a) Hyperplanes, Margins, and Support Vectors

*   **Hyperplane:** The decision boundary that separates different classes in the feature space.
*   **Margin:** The distance between the hyperplane and the nearest data points (support vectors) of any class. SVM aims to maximize this distance.
*   **Support Vectors:** The training data points closest to the hyperplane. They are critical for defining the hyperplane.

### b) Linear SVM

*   For **linearly separable data**, finds a straight hyperplane that maximizes the margin.

### c) The Kernel Trick

*   **Purpose:** Handles **non-linearly separable data**.
*   **Mechanism:** Implicitly maps data into a higher-dimensional feature space where it becomes linearly separable, without explicit computation of new coordinates.
*   **Kernel Functions:** Calculate the dot product in the higher-dimensional space directly from original coordinates.
    *   **Polynomial Kernel:** `K(x, y) = (γ * (xᵀy) + r)ᵈ`
    *   **Radial Basis Function (RBF) / Gaussian Kernel:** `K(x, y) = exp(-γ * ||x - y||²)`
    *   **Sigmoid Kernel:** `K(x, y) = tanh(γ * (xᵀy) + r)`

### d) Advantages of SVMs

*   Effective in high-dimensional spaces.
*   Memory efficient (uses only support vectors).
*   Versatile with different kernel functions.

### e) Disadvantages of SVMs

*   Can be computationally expensive for very large datasets.
*   Sensitive to choice of kernel and hyperparameters.
*   Does not directly provide probability estimates.

---

## 3. Comparison: Bagging vs. Boosting

| Feature       | Bagging (e.g., Random Forest)                               | Boosting (e.g., AdaBoost, Gradient Boosting)                               |
| :------------ | :---------------------------------------------------------- | :------------------------------------------------------------------------- |
| **Training**  | Parallel (models trained independently)                     | Sequential (models trained iteratively, each correcting previous errors)   |
| **Goal**      | Reduces variance (by averaging diverse models)              | Reduces bias (by focusing on hard examples)                                |
| **Weak Learners** | Often deep, complex trees (high variance, low bias)         | Often simple, shallow trees (decision stumps)                              |
| **Data Weighting** | Each model sees a different bootstrap sample (equal weighting within sample) | Data is re-weighted or residuals are modeled (focus on misclassified samples) |
| **Final Model** | Simple average/majority vote of predictions                 | Weighted average of predictions                                            |
