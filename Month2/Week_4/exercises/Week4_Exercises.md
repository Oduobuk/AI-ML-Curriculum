# Month 2, Week 4 Exercises: Boosting & Support Vector Machines (SVM)

## Objective

These exercises are designed to reinforce your understanding of Boosting algorithms and Support Vector Machines (SVMs), their underlying principles, and their application to classification problems.

---

### Exercise 1: Boosting (Conceptual)

1.  **Sequential Learning:** How does Boosting differ from Bagging (e.g., Random Forests) in terms of how individual models are trained? What is the primary advantage of this sequential approach?
2.  **Weak Learners:** Why does Boosting typically use "weak learners" (e.g., shallow decision trees) rather than complex, fully grown trees?
3.  **AdaBoost vs. Gradient Boosting:** Briefly explain the fundamental difference in how AdaBoost and Gradient Boosting iteratively improve their models.

---

### Exercise 2: Support Vector Machines (Conceptual)

1.  **Linearly Separable Data:**
    *   Imagine a 2D dataset with two classes that are perfectly linearly separable. How would an SVM find the optimal decision boundary?
    *   What are "support vectors" in this context, and why are they important?
2.  **Non-Linearly Separable Data:**
    *   Explain the "kernel trick" in SVMs. How does it allow SVMs to classify data that is not linearly separable in its original feature space?
    *   Provide an example of a real-world scenario where a non-linear kernel (like RBF) would likely be more effective than a linear kernel.

---

### Exercise 3: Hyperparameter Tuning for SVM

Consider an SVM with an RBF kernel. What are two key hyperparameters you would need to tune, and what does each parameter generally control?

---

### Exercise 4: Model Selection (Boosting vs. SVM)

You are working on a classification problem with a large, complex tabular dataset (many features, many samples). You need a model that is highly accurate and can handle a mix of numerical and categorical features.

1.  Would you initially lean towards a Boosting algorithm (like XGBoost) or an SVM? Justify your choice.
2.  What are some potential challenges you might face if you chose the other algorithm for this type of dataset?

---

### Exercise 5: True or False

Determine if the following statements are True or False. Justify your answer briefly.

1.  Boosting algorithms train individual models in parallel.
2.  The primary goal of SVM is to minimize the misclassification error on the training data.
3.  The kernel trick explicitly transforms data into a higher-dimensional space.
4.  XGBoost is an implementation of the AdaBoost algorithm.
