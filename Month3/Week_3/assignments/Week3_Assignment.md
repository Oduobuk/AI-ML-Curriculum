# Month 3, Week 3 Assignment: Model Validation and Hyperparameter Tuning

## Objective

This assignment aims to solidify your understanding of model validation techniques (K-Fold Cross-Validation) and hyperparameter tuning strategies (Grid Search or Random Search). You will apply these methods to a classification model, analyze its performance, and discuss the bias-variance trade-off.

## Instructions

1.  **Create a new Python file:** Name it `model_tuning_validation.py`.

2.  **Choose a Dataset:**
    *   Use a built-in `scikit-learn` dataset suitable for classification, such as `load_breast_cancer` or `load_iris`.
    *   Alternatively, you can use the Pima Indians Diabetes dataset.

3.  **Data Preprocessing:**
    *   Load your chosen dataset.
    *   Separate features (X) and target (y).
    *   Split the data into training and testing sets (e.g., 80% train, 20% test).
    *   Scale your features using `StandardScaler` from `sklearn.preprocessing`.

4.  **Choose a Base Model:**
    *   Select a classification model from `scikit-learn` that has hyperparameters to tune. Good choices include:
        *   `LogisticRegression` from `sklearn.linear_model` (e.g., `C` parameter)
        *   `KNeighborsClassifier` from `sklearn.neighbors` (e.g., `n_neighbors`)
        *   `DecisionTreeClassifier` from `sklearn.tree` (e.g., `max_depth`, `min_samples_leaf`)
        *   `RandomForestClassifier` from `sklearn.ensemble` (e.g., `n_estimators`, `max_depth`)

5.  **Perform K-Fold Cross-Validation:**
    *   Import `KFold` (or `StratifiedKFold` for classification) from `sklearn.model_selection`.
    *   Import `cross_val_score` from `sklearn.model_selection`.
    *   Initialize your chosen model with some default hyperparameters.
    *   Initialize `KFold` (e.g., `n_splits=5`, `shuffle=True`, `random_state=42`).
    *   Calculate cross-validation scores using `cross_val_score`.
    *   Print the mean and standard deviation of the cross-validation scores.

6.  **Perform Hyperparameter Tuning (Grid Search or Random Search):**
    *   **Option A: Grid Search**
        *   Import `GridSearchCV` from `sklearn.model_selection`.
        *   Define a dictionary `param_grid` with hyperparameters and their values to search (e.g., for `RandomForestClassifier`: `{'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}`).
        *   Initialize `GridSearchCV` with your model, `param_grid`, `cv` (e.g., 5), and `scoring` (e.g., 'accuracy').
        *   Fit `GridSearchCV` to your *training* data.
        *   Print the `best_params_` and `best_score_`.
    *   **Option B: Random Search**
        *   Import `RandomizedSearchCV` from `sklearn.model_selection`.
        *   Define a dictionary `param_distributions` with hyperparameters and their distributions/values (e.g., for `RandomForestClassifier`: `{'n_estimators': randint(50, 200), 'max_depth': [None, 10, 20]}`).
        *   Initialize `RandomizedSearchCV` with your model, `param_distributions`, `n_iter` (e.g., 10), `cv` (e.g., 5), and `scoring`.
        *   Fit `RandomizedSearchCV` to your *training* data.
        *   Print the `best_params_` and `best_score_`.

7.  **Evaluate Best Model on Test Set:**
    *   Get the `best_estimator_` from your Grid Search or Random Search.
    *   Evaluate this best model on the *unseen test set* using appropriate metrics (e.g., `accuracy_score`, `classification_report`).

8.  **Discussion (in comments or print statements):**
    *   Compare the cross-validation score of the default model with the best score from hyperparameter tuning.
    *   Discuss how hyperparameter tuning improved (or didn't improve) the model.
    *   Relate your findings to the bias-variance trade-off. Did tuning help reduce overfitting or underfitting?

## Example Output (conceptual)

```
--- Initial Model Cross-Validation ---
Mean CV Score: 0.85 (+/- 0.02)

--- Hyperparameter Tuning Results (Grid Search) ---
Best Parameters: {'n_estimators': 100, 'max_depth': 10}
Best CV Score: 0.91

--- Best Model Performance on Test Set ---
Test Set Accuracy: 0.90
Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.93      0.92       150
           1       0.88      0.87      0.87       100

    accuracy                           0.90       250
   macro avg       0.90      0.90      0.90       250
weighted avg       0.90      0.90      0.90       250

Discussion:
- Hyperparameter tuning significantly improved the model's performance from 0.85 to 0.91 (CV score).
- The tuned model achieved 0.90 accuracy on the unseen test set, indicating good generalization.
- This suggests that the default hyperparameters might have led to a slightly underfit model (higher bias), and tuning helped find a better balance, reducing bias without significantly increasing variance.
```

## Submission

*   Save your Python script as `model_tuning_validation.py`.
*   Ensure your code is well-commented and easy to understand.
*   Include your discussion within the script.
