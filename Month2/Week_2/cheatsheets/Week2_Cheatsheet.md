# Month 2, Week 2 Cheatsheet: Logistic Regression & KNN

## 1. Logistic Regression

*   **Purpose:** A **classification algorithm** used to predict the probability of a binary outcome (or multi-class, with extensions).
*   **Output:** Probability `P(Y=1|X)` between 0 and 1.
*   **Core Equation:** `P(Y=1|X) = 1 / (1 + e^(-z))` where `z = β₀ + β₁X₁ + ... + βₚXₚ`
    *   `z`: Linear combination of features (similar to linear regression).
    *   `1 / (1 + e^(-z))`: The **Sigmoid Function**, which squashes `z` to a value between 0 and 1.
*   **Decision Boundary:** A threshold (commonly 0.5) is applied to the probability to classify the outcome.
    *   If `P(Y=1|X) > threshold`, predict class 1.
    *   Otherwise, predict class 0.
*   **Model Fitting:** Typically uses **Maximum Likelihood Estimation (MLE)**, often optimized via Gradient Descent, to find parameters (`β`s) that maximize the likelihood of observing the data.
*   **Cost Function:** Log Loss (or Cross-Entropy Loss) is commonly used.

---

## 2. K-Nearest Neighbors (KNN)

*   **Purpose:** A simple, non-parametric, instance-based **classification (and regression)** algorithm.
*   **"Lazy Learner":** No explicit model training phase; it memorizes the training data.
*   **How it Works (Classification):**
    1.  For a new data point, calculate its distance to all training data points.
    2.  Identify the `K` nearest neighbors.
    3.  Assign the new data point the class label that is most frequent among its `K` neighbors (majority vote).
*   **Distance Metric:** Commonly Euclidean distance, but others like Manhattan distance can be used.
*   **Choosing K:**
    *   Small `K`: More sensitive to noise/outliers, complex decision boundaries.
    *   Large `K`: Smoother decision boundaries, less sensitive to noise, but can miss local patterns.
    *   Often chosen via cross-validation. Odd `K` is preferred for binary classification to avoid ties.
*   **Feature Scaling:** Important for KNN, as distance calculations are sensitive to feature scales.

---

## 3. Evaluation Metrics for Classification

### a) Confusion Matrix
A table summarizing actual vs. predicted classifications:

|                | Predicted Negative | Predicted Positive |
| :------------- | :----------------- | :----------------- |
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

### b) Key Metrics

*   **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`
    *   Overall proportion of correct predictions. Good for balanced datasets.
*   **Precision:** `TP / (TP + FP)`
    *   Of all predicted positives, how many were actually positive? (Minimizes False Positives).
*   **Recall (Sensitivity, True Positive Rate - TPR):** `TP / (TP + FN)`
    *   Of all actual positives, how many were correctly identified? (Minimizes False Negatives).
*   **Specificity (True Negative Rate - TNR):** `TN / (TN + FP)`
    *   Of all actual negatives, how many were correctly identified?
*   **False Positive Rate (FPR):** `FP / (FP + TN)` or `1 - Specificity`
    *   Of all actual negatives, how many were incorrectly identified as positive?
*   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)`
    *   Harmonic mean of Precision and Recall. Useful for imbalanced datasets.

### c) ROC Curve and AUC

*   **ROC (Receiver Operating Characteristic) Curve:**
    *   Plots **TPR (Recall)** against **FPR (1 - Specificity)** at various classification thresholds.
    *   Visualizes the trade-off between true positives and false positives.
    *   A curve closer to the top-left corner indicates better performance.
*   **AUC (Area Under the ROC Curve):**
    *   A single scalar value summarizing the overall performance of a binary classifier across all possible thresholds.
    *   Ranges from 0 to 1.
    *   `AUC = 0.5`: Random classifier.
    *   `AUC = 1.0`: Perfect classifier.
    *   Higher AUC indicates better model discrimination.

---

## 4. Comparison: Linear vs. Logistic Regression

| Feature             | Linear Regression           | Logistic Regression         |
| :------------------ | :-------------------------- | :-------------------------- |
| **Output Type**     | Continuous numerical value  | Probability (0 to 1)        |
| **Task**            | Regression                  | Classification              |
| **Output Function** | Identity (linear equation)  | Sigmoid function            |
| **Cost Function**   | Mean Squared Error (MSE)    | Log Loss (Cross-Entropy)    |
| **Model Fitting**   | Least Squares               | Maximum Likelihood Estimation |
