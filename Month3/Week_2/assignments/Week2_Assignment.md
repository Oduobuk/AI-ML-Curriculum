# Month 3, Week 2 Assignment: Comparing Dimensionality Reduction Techniques

## Objective

This assignment aims to solidify your understanding of different dimensionality reduction techniques: PCA, t-SNE, and UMAP. You will apply these methods to a high-dimensional dataset, visualize the results in 2D, and compare how each method preserves the data's structure.

## Instructions

1.  **Create a new Python file:** Name it `dimensionality_reduction_comparison.py`.

2.  **Load a High-Dimensional Dataset:**
    *   Use a built-in `scikit-learn` dataset: `load_digits` (a smaller version of MNIST, 8x8 pixel images, 64 features).
    *   Alternatively, you could use `load_iris` and artificially increase its dimensionality by adding polynomial features or random noise, but `load_digits` is a more natural high-dimensional example.

3.  **Data Preprocessing:**
    *   Load the dataset.
    *   Separate features (X) and target (y) (the target labels will be used for coloring the plots, not for the reduction itself).
    *   **Feature Scaling:** Apply `StandardScaler` from `sklearn.preprocessing` to your features. This is crucial for PCA and generally beneficial for t-SNE and UMAP.

4.  **Apply PCA:**
    *   Import `PCA` from `sklearn.decomposition`.
    *   Initialize `PCA` with `n_components=2`.
    *   Fit `PCA` to your scaled data and transform it to 2D.
    *   (Optional) Calculate and print the explained variance ratio for each component.

5.  **Apply t-SNE:**
    *   Import `TSNE` from `sklearn.manifold`.
    *   Initialize `TSNE` with `n_components=2`, `random_state` for reproducibility, and `perplexity` (e.g., 30).
    *   Fit `TSNE` to your scaled data and transform it to 2D. (Note: t-SNE's `fit_transform` is often used directly).

6.  **Apply UMAP:**
    *   Install UMAP if you haven't already: `pip install umap-learn`.
    *   Import `UMAP` from `umap`.
    *   Initialize `UMAP` with `n_components=2`, `random_state` for reproducibility, and `n_neighbors` (e.g., 15) and `min_dist` (e.g., 0.1).
    *   Fit `UMAP` to your scaled data and transform it to 2D.

7.  **Visualize and Compare:**
    *   For each dimensionality reduction method (PCA, t-SNE, UMAP), create a scatter plot of the 2D transformed data.
    *   Color-code the points in each scatter plot using the original target labels (`y`).
    *   Use `matplotlib.pyplot` for plotting.
    *   Add titles to each plot indicating the method used.
    *   **Discussion (in comments or print statements):**
        *   Compare the visualizations: Which method produced the clearest separation of clusters?
        *   Which method seemed to preserve the global structure better? Which preserved local structure better?
        *   Comment on the relative computational time (if noticeable) for each method.

## Example Output (conceptual)

```
# PCA Plot (displayed)
# t-SNE Plot (displayed)
# UMAP Plot (displayed)

Discussion:
- PCA showed some separation but clusters were often overlapping, indicating it primarily captured global variance.
- t-SNE produced very distinct, well-separated clusters, highlighting local relationships effectively.
- UMAP also showed clear clusters, often with better preservation of the overall shape of the data compared to t-SNE, and was noticeably faster.
```

## Submission

*   Save your Python script as `dimensionality_reduction_comparison.py`.
*   Ensure your code is well-commented and easy to understand.
*   Include your discussion on the comparison of methods within the script.
