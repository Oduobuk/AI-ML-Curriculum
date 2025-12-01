# Month 3, Week 2: Dimensionality Reduction

## 1. Introduction to Dimensionality Reduction

**Dimensionality Reduction** is a technique used to reduce the number of features (dimensions) in a dataset while retaining as much of the important information as possible. High-dimensional data often suffers from the "curse of dimensionality," which can lead to several problems:

*   **Increased Computational Cost:** More features mean more calculations, leading to slower training times.
*   **Increased Memory Usage:** Storing high-dimensional data requires more memory.
*   **Difficulty in Visualization:** It's hard to visualize data beyond 3 dimensions.
*   **Overfitting:** In high-dimensional spaces, models can easily find spurious correlations, leading to poor generalization.
*   **Sparse Data:** Data points become very sparse in high-dimensional spaces, making it difficult for algorithms to find meaningful patterns.

Dimensionality reduction techniques can be broadly categorized into:
*   **Feature Selection:** Choosing a subset of the original features.
*   **Feature Extraction:** Transforming the data into a new, lower-dimensional feature space.

## 2. Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a linear dimensionality reduction technique that transforms the data into a new coordinate system. The new axes (principal components) are orthogonal and ordered by the amount of variance they capture from the original data.

### a) How PCA Works:
1.  **Center the Data:** Subtract the mean of each feature from all data points for that feature. This centers the data around the origin.
2.  **Calculate Covariance Matrix:** Compute the covariance matrix of the centered data. This matrix describes the relationships between different features.
3.  **Compute Eigenvectors and Eigenvalues:**
    *   **Eigenvectors:** These represent the directions (axes) of the principal components. They indicate the directions of maximum variance in the data.
    *   **Eigenvalues:** These represent the magnitude of variance along each eigenvector. A larger eigenvalue means more variance is captured by that principal component.
4.  **Select Principal Components:** Order the eigenvectors by their corresponding eigenvalues in descending order. Choose the top `k` eigenvectors (where `k` is the desired number of dimensions) to form the new feature subspace.
5.  **Project Data:** Project the original (centered) data onto the selected `k` principal components to obtain the new, lower-dimensional representation.

### b) Key Concepts:
*   **Principal Components (PCs):** The new orthogonal axes. PC1 captures the most variance, PC2 the second most, and so on.
*   **Loading Scores (Eigenvectors):** The coefficients that define how much each original feature contributes to each principal component. They indicate the direction of the principal components.
*   **Eigenvalues:** Quantify the amount of variance explained by each principal component.
*   **Scree Plot:** A plot of the eigenvalues (or percentage of variance explained) against the number of principal components. It helps determine how many principal components to retain by looking for an "elbow" where the explained variance drops significantly.

### c) Applications:
*   Data visualization (reducing to 2 or 3 dimensions).
*   Noise reduction.
*   Feature extraction for other machine learning models.

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**t-SNE** is a non-linear dimensionality reduction technique particularly well-suited for visualizing high-dimensional datasets. It focuses on preserving the local structure of the data.

### a) How t-SNE Works:
1.  **High-Dimensional Probabilities:** It constructs a probability distribution over pairs of high-dimensional objects such that similar objects have a high probability of being picked, while dissimilar points have a low probability. This is often done using a Gaussian distribution.
2.  **Low-Dimensional Probabilities:** It then creates a similar probability distribution over the data points in the low-dimensional map (e.g., 2D or 3D), typically using a Student's t-distribution.
3.  **Minimize Divergence:** t-SNE iteratively adjusts the positions of the points in the low-dimensional map to minimize the Kullback-Leibler (KL) divergence between the high-dimensional and low-dimensional probability distributions. This means it tries to make the low-dimensional representation reflect the similarities found in the high-dimensional data.

### b) Characteristics:
*   **Local Structure Preservation:** Excels at revealing clusters and local relationships within the data.
*   **Non-linear:** Can uncover complex, non-linear relationships that PCA might miss.
*   **Visualization:** Often produces visually appealing and interpretable clusters.
*   **Computational Cost:** Can be computationally intensive for very large datasets.
*   **Global Structure:** May not always preserve global structures (distances between clusters can be misleading).

## 4. UMAP (Uniform Manifold Approximation and Projection)

**UMAP** is another non-linear dimensionality reduction technique that is often faster and more scalable than t-SNE, while frequently producing comparable or superior visualizations. It is based on manifold learning.

### a) How UMAP Works:
1.  **High-Dimensional Graph Construction:** UMAP builds a weighted graph in the high-dimensional space, where edges connect nearby points and edge weights represent the strength of the connection (similarity). It uses fuzzy topological structures to represent the manifold.
2.  **Low-Dimensional Graph Optimization:** It then optimizes a low-dimensional graph to be as structurally similar as possible to the high-dimensional graph. This optimization is done by minimizing the cross-entropy between the two graphs.

### b) Characteristics:
*   **Speed and Scalability:** Generally much faster than t-SNE, making it suitable for larger datasets.
*   **Local and Global Structure Preservation:** Often preserves both local and global data structures better than t-SNE.
*   **Theoretical Foundation:** Has a strong theoretical foundation in Riemannian geometry and algebraic topology.
*   **Visualization:** Produces high-quality visualizations, often with clearer separation between clusters.

## 5. Comparison of PCA, t-SNE, and UMAP

| Feature                 | PCA                                     | t-SNE                                       | UMAP                                        |
| :---------------------- | :-------------------------------------- | :------------------------------------------ | :------------------------------------------ |
| **Type**                | Linear                                  | Non-linear                                  | Non-linear                                  |
| **Goal**                | Maximize variance, decorrelate features | Preserve local structure (neighbor distances) | Preserve local & global structure           |
| **Speed**               | Very Fast                               | Slower (especially on large datasets)       | Faster than t-SNE, often comparable to PCA  |
| **Scalability**         | High                                    | Moderate                                    | High                                        |
| **Global Structure**    | Preserves well                          | Poorly preserved (distances between clusters can be misleading) | Generally preserves well                    |
| **Local Structure**     | Preserves if variance is local          | Preserves very well                         | Preserves very well                         |
| **Interpretability**    | Axes (PCs) are interpretable (loading scores) | Less interpretable                          | Less interpretable                          |
| **Primary Use**         | General DR, noise reduction, feature engineering | Visualization of clusters                   | Visualization, general DR                   |

---
## Recommended Reading

*   **Hands-On ML** — PCA & Manifold Learning
*   **The Elements of Statistical Learning** — Dimensionality Reduction
*   **Dimensionality Reduction: A Short Tutorial** (van der Maaten)
