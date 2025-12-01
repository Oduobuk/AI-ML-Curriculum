# Month 2, Week 2 Exercises: Logistic Regression & KNN

## Objective

These exercises are designed to reinforce your understanding of Logistic Regression and K-Nearest Neighbors (KNN) algorithms, their underlying principles, and how to interpret their performance using various classification metrics.

---

### Exercise 1: Understanding the Sigmoid Function

The sigmoid function is defined as `σ(z) = 1 / (1 + e^(-z))`.

1.  Calculate `σ(z)` for the following values of `z`:
    *   `z = 0`
    *   `z = 2`
    *   `z = -2`
2.  What is the range of values that the sigmoid function can output?
3.  How does the output of the sigmoid function relate to probability in Logistic Regression?

---

### Exercise 2: Interpreting a Confusion Matrix

Consider a binary classification model that predicts whether a customer will churn (positive class) or not churn (negative class). The model's performance on a test set is summarized by the following confusion matrix:

|                | Predicted No Churn | Predicted Churn |
| :------------- | :----------------- | :-------------- |
| **Actual No Churn** | 850                | 50              |
| **Actual Churn** | 100                | 200             |

1.  Identify the values for:
    *   True Positives (TP)
    *   True Negatives (TN)
    *   False Positives (FP)
    *   False Negatives (FN)
2.  Calculate the following metrics:
    *   Accuracy
    *   Precision
    *   Recall (Sensitivity)
    *   Specificity
    *   F1-Score
3.  Based on these metrics, would you say this model is good at identifying churning customers? Why or why not?

---

### Exercise 3: K-Nearest Neighbors (Conceptual)

Imagine you have the following 2D data points, where 'A' and 'B' represent two different classes:

*   Point 1: (1, 1), Class A
*   Point 2: (2, 2), Class A
*   Point 3: (3, 1), Class A
*   Point 4: (4, 4), Class B
*   Point 5: (5, 3), Class B

Now, you have a new data point `P = (3, 3)` that you want to classify.

1.  Calculate the Euclidean distance from point `P` to each of the 5 existing data points.
2.  If `K = 1`, what class would point `P` be assigned to?
3.  If `K = 3`, what class would point `P` be assigned to?
4.  If `K = 5`, what class would point `P` be assigned to?
5.  What are the potential advantages and disadvantages of choosing a very small `K` (e.g., `K=1`) versus a larger `K`?

---

### Exercise 4: ROC Curve and AUC (Conceptual)

1.  What does an ROC curve graphically represent? What are the axes of an ROC curve?
2.  If a model's ROC curve passes through the point (0.5, 0.5), what does that imply about its performance?
3.  What does an AUC score of 0.95 indicate about a binary classification model?
4.  Why is AUC often considered a better metric than accuracy for evaluating models on imbalanced datasets?

---

### Exercise 5: True or False

Determine if the following statements are True or False. Justify your answer briefly.

1.  Logistic Regression is a linear model.
2.  KNN is a parametric model because it learns parameters during training.
3.  A high recall score means that the model has a low number of False Positives.
4.  Feature scaling is generally more important for Logistic Regression than for KNN.
