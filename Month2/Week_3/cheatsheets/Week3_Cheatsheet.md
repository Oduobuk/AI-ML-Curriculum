# Month 2, Week 3 Cheatsheet: Decision Trees & Bagging

## 1. Decision Trees

*   **Purpose:** Non-parametric supervised learning algorithm for **classification** and **regression**. Intuitive, rule-based.
*   **Structure:** Tree-like model of decisions.
    *   **Root Node:** Represents the entire dataset.
    *   **Internal Nodes:** Test on a feature (e.g., `feature_X > threshold`). Branches represent outcomes.
    *   **Leaf Nodes:** Terminal nodes, provide the final prediction (class label or value).
*   **How it Works:** Recursively splits data based on feature values to create pure (homogeneous) child nodes.
*   **Prediction:** Traverse from root to leaf based on feature tests.

### a) Building a Decision Tree: Splitting Criteria

*   **Goal:** Find the best feature and threshold to split a node, maximizing the "purity" of child nodes.
*   **Impurity Measures (for Classification):**
    *   **Gini Impurity:** `Gini = 1 - Σ (pᵢ)²`
        *   `pᵢ`: proportion of samples belonging to class `i` in the node.
        *   Measures probability of misclassifying a randomly chosen element.
        *   Lower Gini = higher purity.
    *   **Entropy:** `Entropy = - Σ pᵢ * log₂(pᵢ)`
        *   Measures uncertainty/randomness.
        *   Lower Entropy = higher purity.
*   **Information Gain:** `Information Gain = Impurity(Parent) - Σ (Weighted_Impurity(Child))`
    *   The reduction in impurity achieved by a split. The algorithm chooses the split that maximizes Information Gain.
*   **Greedy Approach:** Selects the best split at each step locally, without backtracking.
*   **Overfitting:** Individual deep trees are prone to overfitting.

---

## 2. Ensemble Methods

*   **Concept:** Combine multiple individual models (weak learners) to create a stronger, more robust model.

### a) Bagging (Bootstrap Aggregating)

*   **Purpose:** Reduces variance and helps prevent overfitting by averaging predictions from multiple models.
*   **Process:**
    1.  **Bootstrap Sampling:** Create `M` new training datasets by sampling `N` data points from the original dataset (size `N`) **with replacement**.
    2.  **Parallel Training:** Train a separate base model (e.g., a deep decision tree) independently on each of the `M` bootstrap samples.
    3.  **Aggregation:**
        *   **Regression:** Average the predictions of all `M` models.
        *   **Classification:** Use majority voting among the `M` models.

---

## 3. Random Forests

*   **Extension of Bagging:** Specifically for decision trees, adding more randomness to decorrelate trees.
*   **How it Works:**
    1.  **Bootstrap Samples:** Each tree is trained on a different bootstrap sample of the data.
    2.  **Random Feature Subset:** At each split in a tree, only a **random subset of features** is considered as candidates for the split. This ensures diversity among trees.
    3.  **Full Growth:** Individual trees are typically grown to maximum depth (or until leaves are pure/min samples) without pruning.
    4.  **Aggregation:** Predictions are aggregated (majority vote for classification, average for regression).

### a) Advantages of Random Forests

*   **High Accuracy:** Often achieve state-of-the-art performance.
*   **Reduced Overfitting:** More robust than single decision trees due to averaging and decorrelation.
*   **Handles High-Dimensional Data:** Performs well with many features.
*   **Feature Importance:** Provides a measure of how important each feature is for prediction.
*   **Handles Missing Values:** Can be adapted to impute missing values.
*   **Parallelizable:** Trees can be grown independently.

### b) Out-of-Bag (OOB) Error

*   **Concept:** For each tree, approximately 1/3 of the original training data is not included in its bootstrap sample (OOB samples).
*   **Use:** These OOB samples can be used to estimate the generalization error of the random forest without needing a separate validation set.
*   **Calculation:** For each OOB sample, a prediction is made by only the trees that did *not* see that sample during training. These predictions are aggregated to calculate the OOB error.

---

## 4. Feature Importance in Random Forests

*   Random Forests can quantify the importance of each feature.
*   **Mechanism:** Typically measured by how much each feature reduces impurity (e.g., Gini impurity) across all trees in the forest. Features that lead to larger reductions in impurity are considered more important.
*   **Benefit:** Helps in understanding the data and for feature selection.
