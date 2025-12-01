# Month 2, Week 4 Assignment: Implementing Boosting and SVM Classifiers

## Objective

This assignment aims to solidify your understanding of Boosting algorithms and Support Vector Machines (SVMs). You will implement one Boosting classifier (e.g., AdaBoost or Gradient Boosting) and an SVM classifier on a classification dataset, comparing their performance and discussing their characteristics.

## Instructions

1.  **Create a new Python file:** Name it `boosting_svm_models.py`.

2.  **Choose a Dataset:**
    *   Use a built-in `scikit-learn` dataset suitable for classification, such as `load_breast_cancer` (binary classification) or `load_digits` (multi-class classification, a smaller version of MNIST).

3.  **Data Preprocessing:**
    *   Load your chosen dataset.
    *   Separate features (X) and target (y).
    *   Split the data into training and testing sets (e.g., 70% train, 30% test).
    *   **Feature Scaling:** It is crucial for SVMs. Apply a `StandardScaler` to your features.

4.  **Implement a Boosting Classifier:**
    *   **Option A: AdaBoost Classifier**
        *   Import `AdaBoostClassifier` from `sklearn.ensemble`.
        *   Initialize with a base estimator (e.g., `DecisionTreeClassifier(max_depth=1)` for decision stumps) and `n_estimators` (e.g., 50).
    *   **Option B: Gradient Boosting Classifier**
        *   Import `GradientBoostingClassifier` from `sklearn.ensemble`.
        *   Initialize with `n_estimators` (e.g., 100) and `learning_rate` (e.g., 0.1).
    *   Train the chosen Boosting model on your scaled training data.
    *   Make predictions on the scaled test set.
    *   Evaluate its performance using a classification report.

5.  **Implement an SVM Classifier:**
    *   Import `SVC` (Support Vector Classifier) from `sklearn.svm`.
    *   Initialize an `SVC` model. Experiment with different `kernel` types (e.g., 'linear', 'rbf', 'poly') and their respective hyperparameters (e.g., `C`, `gamma`, `degree`).
    *   Train the SVM model on your scaled training data.
    *   Make predictions on the scaled test set.
    *   Evaluate its performance using a classification report.

6.  **Compare Models:**
    *   Print the classification reports for both models.
    *   In comments or print statements, discuss:
        *   Which model performed better on your chosen dataset?
        *   What are the potential advantages and disadvantages of each model type (Boosting vs. SVM) in general?
        *   How did hyperparameter choices (e.g., kernel for SVM, `n_estimators` for Boosting) affect performance?

## Example Output (conceptual)

```
--- Boosting Classifier (GradientBoostingClassifier) ---
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.96      0.95       150
           1       0.94      0.93      0.93       100

    accuracy                           0.94       250
   macro avg       0.94      0.94      0.94       250
weighted avg       0.94      0.94      0.94       250

--- SVM Classifier (kernel='rbf') ---
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       150
           1       0.97      0.96      0.96       100

    accuracy                           0.97       250
   macro avg       0.97      0.97      0.97       250
weighted avg       0.97      0.97      0.97       250

Discussion:
- SVM with RBF kernel performed slightly better on this dataset.
- Boosting models are generally good at handling complex relationships and can be very accurate, but can be prone to overfitting if not tuned well.
- SVMs are powerful in high-dimensional spaces and with clear margins, but can be sensitive to kernel choice and scaling.
```

## Submission

*   Save your Python script as `boosting_svm_models.py`.
*   Ensure your code is well-commented and easy to understand.
*   Include your discussion on model comparison within the script.
