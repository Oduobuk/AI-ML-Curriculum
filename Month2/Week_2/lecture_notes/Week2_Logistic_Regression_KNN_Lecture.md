# Month 2, Week 2: Logistic Regression & K-Nearest Neighbors (KNN)

## 1. Introduction to Classification

While linear regression predicts continuous values, many real-world problems require predicting discrete categories or classes. This is the domain of **classification**. For example, predicting if an email is spam or not, if a customer will churn, or if an image contains a cat or a dog.

## 2. Logistic Regression

**Logistic Regression** is a powerful and widely used classification algorithm, despite its name containing "regression." It models the probability that a given input belongs to a particular class.

### a) The Sigmoid Function
Unlike linear regression which directly outputs a continuous value, logistic regression uses the **sigmoid (or logistic) function** to squish the output of a linear equation into a probability between 0 and 1.

`P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X)))`

Where:
*   `P(Y=1|X)`: The probability that the dependent variable `Y` belongs to class 1, given `X`.
*   `β₀ + β₁X`: The linear combination of features (similar to linear regression).
*   `e`: Euler's number.

The sigmoid function maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

### b) Decision Boundary and Threshold
To convert these probabilities into discrete class predictions, a **threshold** is applied.
*   If `P(Y=1|X) > threshold` (e.g., 0.5), predict class 1.
*   Otherwise, predict class 0.
The choice of threshold can be adjusted based on the specific problem's requirements (e.g., minimizing false negatives for a rare disease).

### c) Model Fitting: Maximum Likelihood Estimation
Instead of minimizing the sum of squared residuals (like linear regression), logistic regression models are typically fit using **Maximum Likelihood Estimation (MLE)**. MLE aims to find the model parameters (`β`s) that maximize the likelihood of observing the actual training data. This is often achieved through iterative optimization algorithms like gradient descent.

### d) Applications
*   Spam detection
*   Disease prediction (e.g., presence/absence of a condition)
*   Customer churn prediction
*   Binary classification tasks in general

## 3. K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, instance-based learning algorithm used for both classification and regression, though more commonly for classification. It's considered a "lazy learner" because it doesn't explicitly learn a model during training; instead, it memorizes the training data.

### a) How KNN Works (for Classification)
1.  **Store Training Data:** The algorithm simply stores all available training data points along with their class labels.
2.  **Calculate Distance:** When a new, unlabeled data point needs to be classified, KNN calculates the distance (e.g., Euclidean distance) between this new point and *all* training data points.
3.  **Identify K-Nearest Neighbors:** It then selects the `K` training data points that are closest (most similar) to the new point.
4.  **Vote for Class:** The new data point is assigned the class label that is most frequent among its `K` nearest neighbors (a majority vote).

### b) Choosing the Value of K
The choice of `K` is crucial:
*   **Small K (e.g., K=1):** Can be noisy and sensitive to outliers. Decision boundaries are more complex.
*   **Large K:** Smoothes out decision boundaries, reduces noise, but can obscure local patterns and lead to misclassification if the neighbors are too far away.
*   **Odd K:** Often preferred to avoid ties in binary classification.
The optimal `K` is typically found through cross-validation.

### c) Applications
*   Recommendation systems
*   Image recognition
*   Credit scoring
*   Medical diagnosis

## 4. Evaluation Metrics for Classification

Evaluating classification models requires different metrics than regression models.

### a) Confusion Matrix
A table that summarizes the performance of a classification model on a set of test data. It shows the number of:
*   **True Positives (TP):** Correctly predicted positive cases.
*   **True Negatives (TN):** Correctly predicted negative cases.
*   **False Positives (FP):** Incorrectly predicted positive cases (Type I error).
*   **False Negatives (FN):** Incorrectly predicted negative cases (Type II error).

### b) Key Metrics Derived from Confusion Matrix
*   **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)` - Overall correctness of the model.
*   **Precision:** `TP / (TP + FP)` - Proportion of positive predictions that were actually correct.
*   **Recall (Sensitivity, True Positive Rate - TPR):** `TP / (TP + FN)` - Proportion of actual positive cases that were correctly identified.
*   **Specificity (True Negative Rate - TNR):** `TN / (TN + FP)` - Proportion of actual negative cases that were correctly identified.
*   **False Positive Rate (FPR):** `FP / (FP + TN)` - Proportion of actual negative cases that were incorrectly identified as positive. `FPR = 1 - Specificity`.
*   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)` - Harmonic mean of precision and recall, useful when there's an uneven class distribution.

### c) ROC Curve and AUC
*   **Receiver Operating Characteristic (ROC) Curve:** A plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold settings.
    *   A curve closer to the top-left corner indicates better performance.
    *   The diagonal line represents a random classifier.
*   **Area Under the ROC Curve (AUC):** A single scalar value that summarizes the overall performance of a classification model across all possible classification thresholds.
    *   `AUC` ranges from 0 to 1.
    *   `AUC = 0.5` indicates a random classifier.
    *   `AUC = 1.0` indicates a perfect classifier.
    *   A higher `AUC` generally means a better model.

---
## Recommended Textbooks

*   **ISLR** – Chapter 4 (Classification)
*   **Python Machine Learning** (Sebastian Raschka) – Logistic Regression & KNN
*   **Elements of Statistical Learning** – Classification Models
