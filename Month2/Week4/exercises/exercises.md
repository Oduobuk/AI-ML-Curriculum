# Week 4 Exercises: Support Vector Machines and Model Evaluation

## Exercise 1: SVM Implementation

1. Load the Iris dataset from scikit-learn
2. Split the data into training and testing sets
3. Train an SVM classifier with a linear kernel
4. Visualize the decision boundary for two features
5. Calculate and print the accuracy score

## Exercise 2: Kernel Methods

1. Generate a non-linearly separable dataset using `make_moons`
2. Train SVMs with different kernels (linear, poly, rbf)
3. Visualize the decision boundaries for each kernel
4. Compare their performance using accuracy scores

## Exercise 3: Model Evaluation Metrics

1. Load the Breast Cancer dataset
2. Train a classifier of your choice
3. Calculate and interpret:
   - Confusion matrix
   - Precision, recall, F1-score
   - ROC curve and AUC score
   - Precision-Recall curve

## Exercise 4: Cross-Validation

1. Implement k-fold cross-validation on the Digits dataset
2. Compare the performance of different classifiers
3. Use cross_val_score to get the mean and standard deviation of scores
4. Visualize the results

## Exercise 5: Hyperparameter Tuning

1. Use GridSearchCV to find the best hyperparameters for an SVM
2. Try different values for C, gamma, and kernel
3. Print the best parameters and best score
4. Visualize the effect of changing C and gamma on the decision boundary

## Exercise 6: Handling Class Imbalance

1. Create an imbalanced classification dataset
2. Train a classifier and evaluate its performance
3. Apply different techniques to handle class imbalance:
   - Class weighting
   - SMOTE oversampling
   - Random undersampling
4. Compare the results using appropriate metrics

## Challenge Problem

1. Load a real-world dataset of your choice
2. Perform exploratory data analysis
3. Preprocess the data as needed
4. Train and evaluate multiple classifiers including SVM
5. Use appropriate evaluation metrics for your problem
6. Write a short report summarizing your findings

## Submission Instructions

1. Create a Jupyter notebook with your solutions
2. Include markdown cells explaining your approach
3. Submit your notebook and any additional files by the deadline
4. Be prepared to present your findings in the next session
