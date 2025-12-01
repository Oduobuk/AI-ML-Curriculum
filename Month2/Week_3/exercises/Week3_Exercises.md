# Month 2, Week 3 Exercises: Decision Trees & Random Forests

## Objective

These exercises are designed to reinforce your understanding of Decision Trees, impurity measures, and the principles behind ensemble methods like Bagging and Random Forests.

---

### Exercise 1: Calculating Gini Impurity and Entropy

Consider a node in a decision tree that contains 10 samples:
*   5 samples belong to Class A
*   3 samples belong to Class B
*   2 samples belong to Class C

1.  **Calculate the Gini Impurity** for this node.
2.  **Calculate the Entropy** for this node (use `log2`).

---

### Exercise 2: Information Gain (Conceptual)

Imagine a parent node with 20 samples:
*   10 samples are Class X
*   10 samples are Class Y
    *   (Entropy of Parent = 1.0)

You are considering a split that divides these 20 samples into two child nodes:

*   **Child Node 1:** 8 samples (7 Class X, 1 Class Y)
    *   (Entropy of Child 1 = 0.544)
*   **Child Node 2:** 12 samples (3 Class X, 9 Class Y)
    *   (Entropy of Child 2 = 0.685)

1.  Calculate the **Information Gain** for this split.
2.  What does a positive Information Gain value indicate?

---

### Exercise 3: Decision Tree Traversal

Consider the following simplified decision tree for classifying fruits based on color and size:

```
Is Color = Red?
├───YES (Red Fruits)
│   └───Is Size = Small?
│       ├───YES (Cherry)
│       └───NO (Apple)
└───NO (Non-Red Fruits)
    └───Is Color = Yellow?
        ├───YES (Banana)
        └───NO (Green Fruits)
            └───Is Size = Small?
                ├───YES (Lime)
                └───NO (Watermelon)
```

Classify the following fruits by tracing their path through the tree:

1.  Fruit A: Color = Red, Size = Large
2.  Fruit B: Color = Green, Size = Small
3.  Fruit C: Color = Yellow, Size = Medium

---

### Exercise 4: Random Forests vs. Single Decision Tree

1.  What are the two main sources of randomness introduced in a Random Forest that are not present in a single Decision Tree?
2.  How do these sources of randomness help to improve the performance of a Random Forest compared to a single, unconstrained Decision Tree? (Focus on bias-variance trade-off).
3.  Explain the concept of "Out-of-Bag (OOB) error" and why it's a useful feature of Random Forests.

---

### Exercise 5: Feature Importance

Imagine a Random Forest model trained to predict house prices. It reports the following feature importances:

*   `Square Footage`: 0.45
*   `Number of Bedrooms`: 0.20
*   `Location Score`: 0.18
*   `Year Built`: 0.10
*   `Number of Bathrooms`: 0.07

1.  Which feature is the most important according to this model?
2.  How can this information be useful to a real estate agent or a home buyer?
3.  What does a feature importance of 0.07 for 'Number of Bathrooms' imply compared to 'Square Footage'?
