# Month 2, Week 2 Assignment: Implementing and Evaluating Classification Models

## Objective

This assignment aims to solidify your understanding of classification algorithms (Logistic Regression and K-Nearest Neighbors) and their evaluation metrics. You will implement one of these models (or both) and evaluate its performance on a classification dataset.

## Instructions

1.  **Create a new Python file:** Name it `classification_model.py`.

2.  **Choose a Dataset:**
    *   You can use a built-in `scikit-learn` dataset (e.g., `load_iris` for multi-class, or `load_breast_cancer` for binary classification).
    *   Alternatively, you can use the Titanic dataset (available on Kaggle) for binary classification (predicting survival).

3.  **Data Preprocessing:**
    *   Load your chosen dataset.
    *   Separate features (X) and target (y).
    *   Handle any missing values (if applicable, e.g., for Titanic).
    *   Split the data into training and testing sets (e.g., 70% train, 30% test).
    *   (Optional but recommended) Scale your features, especially for KNN.

4.  **Choose and Implement a Classifier:**
    *   **Option A: Logistic Regression**
        *   Import `LogisticRegression` from `sklearn.linear_model`.
        *   Initialize and train the model on your training data.
    *   **Option B: K-Nearest Neighbors (KNN)**
        *   Import `KNeighborsClassifier` from `sklearn.neighbors`.
        *   Initialize the model (choose a value for `n_neighbors`, e.g., 3 or 5).
        *   Train the model on your training data.

5.  **Make Predictions:**
    *   Use your trained model to make predictions on the test set.

6.  **Evaluate the Model:**
    *   **Confusion Matrix:**
        *   Import `confusion_matrix` from `sklearn.metrics`.
        *   Calculate and print the confusion matrix.
    *   **Classification Report:**
        *   Import `classification_report` from `sklearn.metrics`.
        *   Print the classification report (which includes Precision, Recall, F1-Score, and Accuracy).
    *   **ROC Curve and AUC (for Binary Classification only):**
        *   If your chosen dataset is for binary classification, calculate the predicted probabilities for the positive class.
        *   Import `roc_curve` and `roc_auc_score` from `sklearn.metrics`.
        *   Calculate the `fpr`, `tpr`, and `thresholds` for the ROC curve.
        *   Calculate and print the `ROC AUC Score`.
        *   (Optional) Plot the ROC curve using `matplotlib`.

## Example Output (conceptual)

```
--- Model: Logistic Regression ---
Confusion Matrix:
[[150   1]
 [  5  94]]

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       151
           1       0.99      0.95      0.97        99

    accuracy                           0.97       250
   macro avg       0.98      0.97      0.97       250
weighted avg       0.97      0.97      0.97       250

ROC AUC Score: 0.985
```

## Submission

*   Save your Python script as `classification_model.py`.
*   Ensure your code is well-commented and easy to understand.
*   Clearly indicate which model you chose (Logistic Regression or KNN) and the dataset used.
