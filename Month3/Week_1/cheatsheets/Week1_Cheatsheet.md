# Month 3, Week 1 Cheatsheet: Clustering

## 1. Unsupervised Learning

*   **Goal:** Discover hidden patterns, structures, or relationships in **unlabeled data**.
*   **Contrast with Supervised Learning:** No target variable (`y`) to predict.
*   **Clustering:** A primary unsupervised technique for grouping similar data points.

---

## 2. K-Means Clustering

*   **Purpose:** Partitions `n` data points into `k` distinct, non-overlapping clusters.
*   **Algorithm Steps:**
    1.  **Initialization:** Choose `K` (number of clusters). Randomly place `K` centroids (`μ`).
    2.  **Assignment (Expectation):** Assign each data point to the nearest centroid.
    3.  **Update (Maximization):** Recalculate each centroid as the mean of all points assigned to its cluster.
    4.  **Repeat:** Iterate steps 2-3 until convergence (assignments don't change, or centroids stabilize).
*   **Objective Function (WC-SSE / Inertia):** Minimizes the sum of squared distances between each point and its assigned centroid.
    `WC-SSE = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²`
*   **Distance Metric:** Typically Euclidean distance.
*   **Challenges:**
    *   Requires pre-specifying `K`.
    *   Sensitive to initial centroid placement (run multiple times with different initializations).
    *   Assumes spherical clusters of similar size and density.
    *   Sensitive to outliers.
    *   Requires feature scaling.

---

## 3. Determining the Optimal Number of Clusters (K)

### a) Elbow Method

*   **Process:**
    1.  Run K-Means for a range of `K` values (e.g., 1 to 10).
    2.  Calculate the `WC-SSE` (inertia) for each `K`.
    3.  Plot `WC-SSE` vs. `K`.
*   **Interpretation:** Look for the "elbow" point where the rate of decrease in `WC-SSE` sharply diminishes. This point suggests a good `K`.

### b) Silhouette Score

*   **Purpose:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
*   **Score Range:** -1 to +1.
*   **Interpretation:**
    *   `+1`: Well-separated clusters.
    *   `0`: Overlapping clusters.
    *   `-1`: Misclassified points.
*   **Usage:** Calculate the average silhouette score for different `K` values. The `K` with the highest average score is often considered optimal.

---

## 4. Other Clustering Algorithms (Brief Overview)

### a) Hierarchical Clustering

*   **Concept:** Builds a hierarchy of clusters.
    *   **Agglomerative (Bottom-Up):** Starts with each point as a single cluster, then iteratively merges the closest clusters until all points are in one cluster or a stopping criterion is met.
    *   **Divisive (Top-Down):** Starts with all points in one cluster, then recursively splits clusters.
*   **Dendrogram:** A tree-like diagram showing the hierarchy of clusters.
*   **Distance Metrics:** Can use various linkage criteria (e.g., single, complete, average) to define distance between clusters.

### b) DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

*   **Concept:** Groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.
*   **Key Parameters:**
    *   `eps` (ε): Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    *   `min_samples`: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
*   **Advantages:** Can find arbitrarily shaped clusters, robust to outliers, does not require specifying `K`.
*   **Disadvantages:** Struggles with varying densities, sensitive to parameter choice.

---

## 5. Feature Scaling

*   **Importance:** Crucial for distance-based algorithms like K-Means and Hierarchical Clustering.
*   **Reason:** Features with larger ranges can disproportionately influence distance calculations.
*   **Common Methods:** Standardization (Z-score scaling) or Min-Max Scaling.
