# Month 2, Week 3 Assignment: Implementing Decision Trees and Random Forests

## Objective

This assignment aims to solidify your understanding of Decision Trees and Random Forests. You will implement a Decision Tree classifier and then build a Random Forest classifier, evaluating their performance and exploring feature importance.

## Instructions

1.  **Create a new Python file:** Name it `tree_ensemble_models.py`.

2.  **Choose a Dataset:**
    *   Use a built-in `scikit-learn` dataset suitable for classification, such as `load_iris` or `load_breast_cancer`.
    *   Alternatively, you can use the Pima Indians Diabetes dataset (available online, e.g., UCI Machine Learning Repository) for binary classification.

3.  **Data Preprocessing:**
    *   Load your chosen dataset.
    *   Separate features (X) and target (y).
    *   Split the data into training and testing sets (e.g., 70% train, 30% test).
    *   (Optional) Scale your features, though tree-based models are less sensitive to scaling than distance-based models.

4.  **Implement a Decision Tree Classifier:**
    *   Import `DecisionTreeClassifier` from `sklearn.tree`.
    *   Initialize a `DecisionTreeClassifier`. Experiment with hyperparameters like `max_depth` (e.g., 3, 5, 10) and `criterion` ('gini' or 'entropy').
    *   Train the model on your training data.
    *   Make predictions on the test set.
    *   Evaluate its performance using a classification report (Precision, Recall, F1-Score, Accuracy).
    *   (Optional) Visualize the decision tree using `plot_tree` from `sklearn.tree` (requires `matplotlib`).

5.  **Implement a Random Forest Classifier:**
    *   Import `RandomForestClassifier` from `sklearn.ensemble`.
    *   Initialize a `RandomForestClassifier`. Experiment with hyperparameters like `n_estimators` (number of trees, e.g., 100), `max_depth`, and `max_features` (e.g., 'sqrt', 'log2').
    *   Train the model on your training data.
    *   Make predictions on the test set.
    *   Evaluate its performance using a classification report.
    *   **Extract Feature Importance:**
        *   Access the `feature_importances_` attribute of your trained `RandomForestClassifier`.
        *   Print the feature importances, ideally mapping them back to the original feature names.

6.  **Compare Models:**
    *   Briefly discuss the performance difference between the single Decision Tree and the Random Forest.
    *   Comment on the most important features identified by the Random Forest.

## Example Output (conceptual)

```
--- Decision Tree Classifier (max_depth=5) ---
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.92      0.91       150
           1       0.88      0.85      0.86       100

    accuracy                           0.89       250
   macro avg       0.89      0.89      0.89       250
weighted avg       0.89      0.89      0.89       250

--- Random Forest Classifier (n_estimators=100, max_depth=5) ---
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.96      0.95       150
           1       0.94      0.93      0.93       100

    accuracy                           0.94       250
   macro avg       0.94      0.94      0.94       250
weighted avg       0.94      0.94      0.94       250

Feature Importances:
Feature_A: 0.35
Feature_B: 0.28
Feature_C: 0.15
...
```

## Submission

*   Save your Python script as `tree_ensemble_models.py`.
*   Ensure your code is well-commented and easy to understand.
*   Include your discussion on model comparison and feature importance within the script as comments or print statements.
