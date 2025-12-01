# Month 2, Week 4: Boosting & Support Vector Machines (SVM)

## 1. Boosting

**Boosting** is an ensemble learning technique that combines multiple "weak learners" (models that perform slightly better than random chance) sequentially to form a single "strong learner." Unlike bagging (where models are trained independently), boosting trains models iteratively, with each new model focusing on correcting the errors of the previous ones.

### a) How Boosting Works
1.  **Initial Model:** A weak learner is trained on the original dataset.
2.  **Weighted Data:** Subsequent weak learners are trained on a modified version of the dataset, where samples that were misclassified by previous models are given higher weights. This forces the new models to pay more attention to the "hard" examples.
3.  **Weighted Combination:** The predictions from all weak learners are combined (typically with weights assigned based on their performance) to produce the final prediction.

### b) AdaBoost (Adaptive Boosting)
*   **Mechanism:** AdaBoost starts by assigning equal weights to all training samples. In each iteration, it trains a weak learner (often a shallow decision tree, called a "decision stump"). It then increases the weights of misclassified samples and decreases the weights of correctly classified samples. The final model is a weighted sum of all weak learners.
*   **Focus:** Primarily on reducing bias.

### c) Gradient Boosting
*   **Mechanism:** Instead of re-weighting data, Gradient Boosting builds new models that predict the **residuals (errors)** of the previous models. Each new tree is trained to minimize the loss function by moving in the direction of the negative gradient (hence "gradient" boosting).
*   **Generalization:** Can be used with various loss functions and weak learners.

### d) Advanced Gradient Boosting Implementations
*   **XGBoost (Extreme Gradient Boosting):** A highly optimized, scalable, and efficient implementation of gradient boosting. It includes regularization techniques (L1 and L2), parallel processing, tree pruning, and handling of missing values, making it very popular for competitive machine learning.
*   **LightGBM:** Another gradient boosting framework that uses tree-based learning algorithms, designed for distributed and efficient training.
*   **CatBoost:** A gradient boosting library that excels at handling categorical features automatically and effectively, and uses ordered boosting to combat prediction shift.

## 2. Support Vector Machines (SVM)

**Support Vector Machines (SVMs)** are powerful and versatile supervised learning models used for classification, regression, and outlier detection. They are particularly effective in high-dimensional spaces and when the number of dimensions is greater than the number of samples.

### a) Hyperplanes and Margins
*   **Hyperplane:** In SVM, a hyperplane is the decision boundary that separates different classes in the feature space. In 2D, it's a line; in 3D, it's a plane; in higher dimensions, it's a hyperplane.
*   **Margin:** SVM aims to find the hyperplane that has the largest possible distance to the nearest training data points of any class. This distance is called the margin. A larger margin generally leads to better generalization.
*   **Support Vectors:** The training data points that lie closest to the decision boundary (the hyperplane) are called support vectors. These are the critical points that define the orientation and position of the hyperplane. If these points were removed, the hyperplane would change.

### b) Linear SVM
*   For data that is **linearly separable** (i.e., a single straight line or hyperplane can perfectly separate the classes), a linear SVM finds the optimal hyperplane that maximizes the margin between the classes.

### c) The Kernel Trick: Handling Non-Linear Data
*   Many real-world datasets are **not linearly separable**. To handle such cases, SVMs use the **kernel trick**.
*   **Mechanism:** The kernel trick implicitly maps the original input features into a higher-dimensional feature space where the data *becomes linearly separable*. This transformation is done without explicitly calculating the new, high-dimensional coordinates.
*   **Kernel Functions:** These functions calculate the dot product between two vectors in the higher-dimensional space directly from their original coordinates. Common kernel functions include:
    *   **Polynomial Kernel:** `K(x, y) = (γ * (xᵀy) + r)ᵈ`
    *   **Radial Basis Function (RBF) / Gaussian Kernel:** `K(x, y) = exp(-γ * ||x - y||²)`
    *   **Sigmoid Kernel:** `K(x, y) = tanh(γ * (xᵀy) + r)`
*   **Effect:** The kernel trick allows SVMs to find non-linear decision boundaries in the original feature space by finding a linear hyperplane in the transformed, higher-dimensional space.

### d) Advantages of SVMs
*   Effective in high-dimensional spaces.
*   Memory efficient, as it uses a subset of training points (support vectors) in the decision function.
*   Versatile: Different kernel functions can be specified for the decision function.

### e) Disadvantages of SVMs
*   Can be computationally expensive for very large datasets.
*   Choosing the correct kernel function and tuning its parameters can be challenging.
*   Do not directly provide probability estimates (these are calculated using an expensive cross-validation procedure).

---
## Recommended Textbooks

*   **Hands-On Machine Learning** – SVM & Boosting chapters
*   **Elements of Statistical Learning** – Boosting & SVM chapters
*   **XGBoost Official Documentation**
