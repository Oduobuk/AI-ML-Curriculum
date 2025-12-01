# Month 3, Week 1 Exercises: Clustering

## Objective

These exercises are designed to reinforce your understanding of clustering algorithms, particularly K-Means, and methods for evaluating clustering results.

---

### Exercise 1: K-Means Algorithm Steps (Manual Trace)

Imagine you have the following 1D data points: `[1, 2, 8, 9, 10]`
And you want to perform K-Means clustering with `K = 2`.

Assume the initial centroids are randomly chosen as:
*   Centroid 1 (`μ₁`) = 1
*   Centroid 2 (`μ₂`) = 2

Trace the first two iterations of the K-Means algorithm:

**Iteration 1:**
1.  **Assignment Step:** Assign each data point to the nearest centroid.
    *   Which points belong to Cluster 1?
    *   Which points belong to Cluster 2?
2.  **Update Step:** Recalculate the centroids.
    *   New `μ₁` = ?
    *   New `μ₂` = ?

**Iteration 2:**
1.  **Assignment Step:** Assign each data point to the nearest *new* centroid.
    *   Which points belong to Cluster 1?
    *   Which points belong to Cluster 2?
2.  **Update Step:** Recalculate the centroids.
    *   New `μ₁` = ?
    *   New `μ₂` = ?

Will the algorithm converge after this iteration, or will it continue? Explain why.

---

### Exercise 2: Interpreting an Elbow Plot

You run K-Means clustering on a dataset for `K` values from 1 to 10 and generate the following SSE (Sum of Squared Errors) values:

| K | SSE |
|---|-----|
| 1 | 1500|
| 2 | 600 |
| 3 | 250 |
| 4 | 180 |
| 5 | 150 |
| 6 | 130 |
| 7 | 115 |
| 8 | 105 |
| 9 | 98  |
| 10| 92  |

1.  If you were to plot these values, where would you visually identify the "elbow" point?
2.  Based on the Elbow Method, what would be your suggested optimal number of clusters (`K`)?
3.  Explain the intuition behind why the elbow point is considered optimal.

---

### Exercise 3: Understanding Silhouette Score

1.  What does a Silhouette Score of approximately `0.7` for a data point indicate about its clustering?
2.  What does a Silhouette Score of approximately `-0.2` for a data point suggest?
3.  If the average Silhouette Score for `K=3` is `0.65` and for `K=4` is `0.72`, which `K` would you prefer based on this metric, and why?

---

### Exercise 4: Feature Scaling for Clustering

1.  Why is feature scaling (e.g., using `StandardScaler` or `MinMaxScaler`) generally recommended before applying K-Means clustering?
2.  Provide a simple example of how unscaled features could lead to misleading clustering results.

---

### Exercise 5: K-Means Limitations

1.  K-Means clustering tends to form spherical clusters. How might this be a limitation when trying to cluster data that forms non-spherical shapes (e.g., crescent moons or concentric circles)?
2.  What is one alternative clustering algorithm that might perform better on such non-spherical data?
