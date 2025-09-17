"""
Exercise 1: K-Means Clustering Implementation

In this exercise, you'll implement the K-Means algorithm from scratch
and apply it to a synthetic dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        """
        Initialize the KMeans algorithm.
        
        Parameters:
        - n_clusters: int, number of clusters
        - max_iter: int, maximum number of iterations
        - random_state: int, random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
    def initialize_centroids(self, X):
        """Randomly initialize centroids from data points."""
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids
    
    def fit(self, X):
        """Fit the KMeans model to the data."""
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            labels = pairwise_distances_argmin(X, self.centroids)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) 
                                    for i in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        self.labels_ = pairwise_distances_argmin(X, self.centroids)
        return self
    
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        return pairwise_distances_argmin(X, self.centroids)

def main():
    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.60,
        random_state=42
    )
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels_
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    # Plot true clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
    plt.title("True Clusters")
    plt.axis('equal')
    
    # Plot predicted clusters
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', s=200, alpha=0.7, marker='X')
    plt.title("K-Means Clustering")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png')
    plt.show()

if __name__ == "__main__":
    main()
