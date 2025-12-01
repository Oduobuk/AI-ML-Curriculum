# Month 3, Week 1 Assignment: K-Means Clustering and Evaluation

## Objective

This assignment aims to solidify your understanding of K-Means clustering and methods for evaluating clustering performance. You will implement K-Means (or use `scikit-learn`), determine the optimal number of clusters (K) using the Elbow Method and Silhouette Score, and visualize the results.

## Instructions

1.  **Create a new Python file:** Name it `kmeans_clustering.py`.

2.  **Generate or Load a Dataset:**
    *   **Option A (Recommended for visualization):** Generate a synthetic 2D dataset using `make_blobs` from `sklearn.datasets`. Create a dataset with a known number of clusters (e.g., 3 or 4) to test your methods.
        ```python
        from sklearn.datasets import make_blobs
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        ```
    *   **Option B:** Load a real-world dataset suitable for clustering, such as the Iris dataset (using `load_iris` from `sklearn.datasets`, but *discard the labels* for clustering).

3.  **Data Preprocessing:**
    *   (If using a real-world dataset) Perform any necessary scaling of features using `StandardScaler` from `sklearn.preprocessing`. This is crucial for distance-based algorithms like K-Means.

4.  **Implement K-Means Clustering:**
    *   Import `KMeans` from `sklearn.cluster`.
    *   **Part 1: Elbow Method**
        *   Create an empty list to store the Sum of Squared Errors (SSE) for different K values.
        *   Loop through a range of `K` values (e.g., from 1 to 10).
        *   For each `K`:
            *   Initialize `KMeans` with `n_clusters=K` and `random_state` for reproducibility.
            *   Fit `KMeans` to your data.
            *   Append the `inertia_` attribute (which is the SSE) to your list.
        *   Plot the SSE values against `K` using `matplotlib.pyplot`.
        *   Identify the "elbow" point visually and suggest an optimal `K`.
    *   **Part 2: Silhouette Score**
        *   Create an empty list to store silhouette scores for different K values.
        *   Loop through a range of `K` values (e.g., from 2 to 10, as silhouette score is not defined for K=1).
        *   For each `K`:
            *   Initialize `KMeans` with `n_clusters=K` and `random_state`.
            *   Fit `KMeans` to your data.
            *   Calculate the `silhouette_score` using `silhouette_score` from `sklearn.metrics` (requires `X` and the cluster labels from `kmeans.labels_`).
            *   Append the score to your list.
        *   Plot the silhouette scores against `K`.
        *   Identify the `K` that yields the highest silhouette score and suggest it as the optimal `K`.

5.  **Final Clustering and Visualization:**
    *   Choose your final optimal `K` based on the Elbow Method and Silhouette Score.
    *   Initialize and fit `KMeans` with this optimal `K` to your data.
    *   Get the cluster labels (`kmeans.labels_`) and the final cluster centroids (`kmeans.cluster_centers_`).
    *   **Visualize the clusters:**
        *   Use `matplotlib.pyplot.scatter` to plot your data points.
        *   Color-code the points based on their assigned cluster labels.
        *   Plot the cluster centroids on top of the data points (e.g., using a different marker and color).
        *   Add labels and a title to your plot.

## Example Output (conceptual)

```
# Elbow Method Plot (displayed)
# Silhouette Score Plot (displayed)

Optimal K suggested by Elbow Method: 4
Optimal K suggested by Silhouette Score: 4

--- Final Clustering with K=4 ---
# Scatter plot of clusters and centroids (displayed)
```

## Submission

*   Save your Python script as `kmeans_clustering.py`.
*   Ensure your code is well-commented and easy to understand.
*   Include comments in your code indicating your chosen optimal `K` and the reasoning.
