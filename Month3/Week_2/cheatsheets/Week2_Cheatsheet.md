# Month 3, Week 2 Cheatsheet: Dimensionality Reduction

## 1. Introduction to Dimensionality Reduction

*   **Purpose:** Reduce the number of features (dimensions) in a dataset.
*   **Why?**
    *   **Curse of Dimensionality:** Problems arising in high-dimensional spaces (data sparsity, increased computation, overfitting).
    *   **Visualization:** Easier to plot and understand data in 2D/3D.
    *   **Noise Reduction:** Can remove irrelevant or redundant features.
    *   **Computational Efficiency:** Faster training for subsequent models.
*   **Types:**
    *   **Feature Selection:** Choosing a subset of original features.
    *   **Feature Extraction:** Creating new, lower-dimensional features from combinations of original ones.

---

## 2. Principal Component Analysis (PCA)

*   **Type:** Linear Feature Extraction.
*   **Goal:** Find orthogonal axes (Principal Components - PCs) that capture the maximum variance in the data.
*   **How it Works:**
    1.  **Center Data:** Subtract mean from each feature.
    2.  **Compute Covariance Matrix:** Describes feature relationships.
    3.  **Eigen-decomposition:** Find Eigenvectors (directions of PCs) and Eigenvalues (magnitudes of variance along PCs).
    4.  **Project Data:** Transform data onto the top `k` PCs.
*   **Key Concepts:**
    *   **Principal Components (PCs):** New orthogonal axes, ordered by variance explained.
    *   **Loading Scores (Eigenvectors):** Contribution of original features to each PC.
    *   **Eigenvalues:** Amount of variance explained by each PC.
    *   **Scree Plot:** Visualizes eigenvalues to help select `k` (look for "elbow").
*   **Advantages:** Interpretable axes, fast, preserves global variance.
*   **Limitations:** Linear method, may not capture non-linear relationships.

---

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

*   **Type:** Non-linear Feature Extraction (Manifold Learning).
*   **Goal:** Visualize high-dimensional data by preserving **local structure** (neighbor relationships) in a low-dimensional space (typically 2D/3D).
*   **How it Works:**
    1.  Converts high-dimensional Euclidean distances into conditional probabilities (Gaussian distribution).
    2.  Creates similar probabilities in a low-dimensional space (Student's t-distribution).
    3.  Minimizes the Kullback-Leibler (KL) divergence between these two distributions.
*   **Key Concepts:**
    *   **Perplexity:** Controls the balance between local and global aspects of the data. Roughly, the number of nearest neighbors considered.
    *   **Local Structure:** Focuses on keeping nearby points nearby.
*   **Advantages:** Excellent for visualizing clusters, reveals non-linear structures.
*   **Limitations:** Computationally intensive, doesn't preserve global distances well (distances between clusters can be misleading), sensitive to `perplexity`.

---

## 4. UMAP (Uniform Manifold Approximation and Projection)

*   **Type:** Non-linear Feature Extraction (Manifold Learning).
*   **Goal:** General-purpose dimensionality reduction, often for visualization, preserving both **local and global structure**.
*   **How it Works:**
    1.  Builds a weighted graph in high-dimensional space (fuzzy topological structure).
    2.  Optimizes a low-dimensional graph to be as structurally similar as possible to the high-dimensional graph.
*   **Key Concepts:**
    *   **Manifold Learning:** Assumes data lies on a lower-dimensional manifold embedded in a higher-dimensional space.
    *   **`n_neighbors`:** Controls how UMAP balances local vs. global structure.
    *   **`min_dist`:** Controls how tightly UMAP packs points together.
*   **Advantages:** Faster than t-SNE, better scalability, often preserves global structure better than t-SNE, strong theoretical foundation.
*   **Limitations:** Still non-linear, interpretability of axes is limited.

---

## 5. Comparison

| Feature                 | PCA                                     | t-SNE                                       | UMAP                                        |
| :---------------------- | :-------------------------------------- | :------------------------------------------ | :------------------------------------------ |
| **Linear/Non-linear**   | Linear                                  | Non-linear                                  | Non-linear                                  |
| **Speed**               | Very Fast                               | Slower                                      | Fast                                        |
| **Scalability**         | High                                    | Moderate                                    | High                                        |
| **Global Structure**    | Preserves well                          | Poorly preserved (distances between clusters can be misleading) | Generally preserves well                    |
| **Local Structure**     | Preserves if variance is local          | Preserves very well                         | Preserves very well                         |
| **Interpretability**    | Axes (PCs) are interpretable (loading scores) | Less interpretable                          | Less interpretable                          |
| **Primary Use**         | General DR, noise reduction, feature engineering | Visualization of clusters                   | Visualization, general DR                   |