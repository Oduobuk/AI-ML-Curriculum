# Month 3, Week 2 Exercises: Dimensionality Reduction

## Objective

These exercises are designed to reinforce your understanding of dimensionality reduction techniques, including PCA, t-SNE, and UMAP, and their applications.

---

### Exercise 1: PCA (Conceptual)

Imagine you have a dataset with 100 features. After performing PCA, you get the following explained variance ratios for the first few principal components:

*   PC1: 0.40
*   PC2: 0.25
*   PC3: 0.10
*   PC4: 0.05

1.  What percentage of the total variance in the dataset is explained by the first two principal components (PC1 and PC2)?
2.  If you wanted to reduce the dimensionality to 2 components, what information would you be losing compared to using all 100 features?
3.  How would you interpret a loading score of `0.8` for 'Feature_A' on PC1, and `-0.1` for 'Feature_B' on PC1?

---

### Exercise 2: Curse of Dimensionality

1.  Briefly explain what the "curse of dimensionality" refers to in machine learning.
2.  How does dimensionality reduction help mitigate the problems associated with high-dimensional data?

---

### Exercise 3: Linear vs. Non-Linear Dimensionality Reduction

You have a dataset where data points belonging to different classes form concentric circles in a 2D feature space.

1.  Would PCA likely be effective in separating these classes if you reduced the dimensionality to 1? Why or why not?
2.  Which non-linear dimensionality reduction technique (t-SNE or UMAP) would you expect to perform better for visualizing these concentric circles in 2D, and why?

---

### Exercise 4: t-SNE and UMAP (Conceptual)

1.  What is the primary difference in the objective of t-SNE and UMAP compared to PCA when it comes to preserving data structure?
2.  If you observe tightly packed clusters in a t-SNE plot, can you confidently say that the points within those clusters are very close in the original high-dimensional space? What about the distances *between* clusters?
3.  What are two advantages of UMAP over t-SNE?

---

### Exercise 5: Feature Selection vs. Feature Extraction

1.  Distinguish between "feature selection" and "feature extraction" in the context of dimensionality reduction.
2.  Provide an example of a scenario where feature selection might be preferred over feature extraction.