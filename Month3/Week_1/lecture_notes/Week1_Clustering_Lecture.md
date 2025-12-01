# Month 3, Week 1: Clustering

## 1. Introduction to Unsupervised Learning and Clustering

In contrast to supervised learning (where models learn from labeled data), **unsupervised learning** deals with unlabeled data. The goal is to discover hidden patterns, structures, or relationships within the data without explicit guidance.

**Clustering** is a prominent unsupervised learning technique that aims to group a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups.

### Applications of Clustering:
*   **Customer Segmentation:** Grouping customers with similar purchasing behaviors.
*   **Anomaly Detection:** Identifying unusual data points that don't fit into any cluster.
*   **Document Analysis:** Grouping similar articles or documents.
*   **Image Segmentation:** Dividing an image into regions of similar pixels.
*   **Genomic Analysis:** Grouping genes with similar expression patterns.

## 2. K-Means Clustering

**K-Means** is one of the most popular and widely used partitioning clustering algorithms. It's an iterative algorithm that aims to partition `n` data points into `k` clusters, where each data point belongs to the cluster with the nearest mean (centroid).

### a) The K-Means Algorithm Steps:
1.  **Initialization:**
    *   Choose the number of clusters, `K`.
    *   Randomly initialize `K` cluster centroids (mean vectors) in the data space. These centroids can be randomly selected data points or random points within the data's range.
2.  **Assignment Step (Expectation):**
    *   Assign each data point to the cluster whose centroid is closest (e.g., using Euclidean distance).
3.  **Update Step (Maximization):**
    *   Recalculate the position of each cluster centroid by taking the mean of all data points assigned to that cluster.
4.  **Iteration:**
    *   Repeat steps 2 and 3 until the cluster assignments no longer change, or the centroids' positions stabilize, or a maximum number of iterations is reached.

### b) K-Means Objective Function (Cost Function)
K-Means aims to minimize the **Within-Cluster Sum of Squared Errors (WC-SSE)**, also known as inertia. This measures the sum of squared distances between each data point and its assigned cluster centroid.
`WC-SSE = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²`
Where `Cᵢ` is the `i`-th cluster and `μᵢ` is its centroid.

### c) Challenges with K-Means:
*   **Sensitivity to Initial Centroids:** The final clustering result can depend heavily on the initial placement of centroids. Running the algorithm multiple times with different random initializations and choosing the best result (lowest WC-SSE) is common practice.
*   **Requires Pre-specifying K:** The number of clusters `K` must be chosen beforehand.
*   **Assumes Spherical Clusters:** K-Means tends to form spherical clusters and struggles with clusters of irregular shapes or varying densities.
*   **Sensitivity to Outliers:** Outliers can significantly affect cluster centroids.

## 3. Determining the Optimal Number of Clusters (K)

Choosing the right `K` is crucial for effective clustering.

### a) The Elbow Method
The Elbow Method is a heuristic used to determine the optimal `K`.
1.  Run K-Means for a range of `K` values (e.g., from 1 to 10).
2.  For each `K`, calculate the WC-SSE (inertia).
3.  Plot the WC-SSE against `K`.
4.  Look for an "elbow" point in the plot, where the rate of decrease in WC-SSE sharply changes. This point often indicates a good balance between minimizing error and avoiding too many clusters.

### b) Silhouette Score
The Silhouette Score is a metric used to evaluate the quality of clusters. It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
*   **Score Range:** -1 to +1.
*   **Interpretation:**
    *   **+1:** Indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
    *   **0:** Indicates that the object is on or very close to the decision boundary between two neighboring clusters.
    *   **-1:** Indicates that the object is probably assigned to the wrong cluster.
*   **Usage:** Calculate the average silhouette score for different `K` values. The `K` that yields the highest average silhouette score is often considered optimal.

## 4. Other Clustering Algorithms (Brief Mention)

While K-Means is popular, other algorithms exist for different data characteristics:
*   **Hierarchical Clustering:** Builds a hierarchy of clusters (dendrogram) either by starting with individual points and merging them (agglomerative) or starting with one large cluster and splitting it (divisive).
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Identifies clusters based on density. It can find arbitrarily shaped clusters and is robust to outliers. It requires two parameters: `epsilon` (maximum distance between two samples for one to be considered as in the neighborhood of the other) and `min_samples` (the number of samples in a neighborhood for a point to be considered as a core point).

---
## Recommended Reading

*   **Hands-On Machine Learning** (Aurélien Géron) — Clustering chapter
*   **Introduction to Statistical Learning (ISLR)** — Unsupervised Learning
*   **Pattern Recognition & Machine Learning** (Bishop) — Clustering
