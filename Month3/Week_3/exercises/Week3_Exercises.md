# Month 3, Week 3 Exercises: Model Validation & Hyperparameter Tuning

## Objective

These exercises are designed to reinforce your understanding of model validation techniques, hyperparameter tuning strategies, and the bias-variance trade-off.

---

### Exercise 1: Bias-Variance Trade-off (Conceptual)

1.  **Underfitting:** Describe a scenario where a machine learning model would likely suffer from underfitting. What are typical characteristics of an underfit model's performance on training and test data?
2.  **Overfitting:** Describe a scenario where a machine learning model would likely suffer from overfitting. What are typical characteristics of an overfit model's performance on training and test data?
3.  How does increasing the complexity of a model generally affect its bias and variance?

---

### Exercise 2: Cross-Validation (Conceptual)

You are training a classification model on a dataset of 1000 samples.

1.  **Train-Test Split:** If you use a simple 80/20 train-test split, how many samples are used for training and how many for testing? What is a potential drawback of this method compared to cross-validation?
2.  **K-Fold Cross-Validation:** If you use 5-Fold Cross-Validation:
    *   How many times will the model be trained?
    *   How many samples will be in the training set and validation set for each fold?
    *   What is the main advantage of using K-Fold Cross-Validation over a single train-test split?
3.  When would you prefer to use **Stratified K-Fold Cross-Validation**?

---

### Exercise 3: Hyperparameter Tuning Strategies

1.  **Grid Search:** You are tuning a Random Forest Classifier with `n_estimators = [50, 100, 200]` and `max_depth = [5, 10, None]`. How many unique models will Grid Search train and evaluate?
2.  **Random Search:** If you use Random Search with `n_iter = 10` for the same Random Forest hyperparameters, how many unique models will be trained? What is a potential advantage of Random Search over Grid Search in this scenario?
3.  What is the primary goal of hyperparameter tuning?

---

### Exercise 4: Interpreting MLflow Tracking

Imagine you are using MLflow Tracking for your experiments. For a particular model, you log the following:

*   `parameters`: `learning_rate=0.01`, `n_estimators=100`
*   `metrics`: `accuracy=0.88`, `f1_score=0.85`
*   `artifact`: `model.pkl`

1.  What information does MLflow Tracking allow you to easily compare across different runs of your model?
2.  How does logging the `model.pkl` as an artifact contribute to reproducibility?

---

### Exercise 5: True or False

Determine if the following statements are True or False. Justify your answer briefly.

1.  A model with high variance is typically underfitting the training data.
2.  Cross-validation is primarily used to prevent overfitting during the training phase.
3.  Hyperparameters are learned by the model during the training process.
4.  MLflow is a machine learning algorithm.
