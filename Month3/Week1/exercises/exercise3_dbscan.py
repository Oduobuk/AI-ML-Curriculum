"""
Exercise 3: DBSCAN Clustering

In this exercise, you'll work with DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
to identify clusters in data with noise and varying densities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def plot_clusters(X, y_pred, title):
    """Visualize clustering results with noise points in black."""
    plt.figure(figsize=(10, 6))
    
    # Plot noise points first (label = -1)
    noise_mask = (y_pred == -1)
    if np.any(noise_mask):
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                   c='black', marker='x', alpha=0.5, label='Noise')
    
    # Plot clusters
    for cluster_id in np.unique(y_pred[y_pred != -1]):
        cluster_mask = (y_pred == cluster_id)
        plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                   label=f'Cluster {cluster_id}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def find_optimal_eps(X, min_samples, k=4):
    """Find optimal epsilon using k-distance graph."""
    # Calculate distances to k nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    
    # Sort distances
    distances = np.sort(distances[:, -1])
    
    # Find the elbow point
    from kneed import KneeLocator
    kneedle = KneeLocator(
        range(len(distances)), distances, 
        S=1.0, curve='convex', direction='increasing'
    )
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances, 'b-', label='k-distance')
    if kneedle.elbow is not None:
        plt.axhline(y=distances[kneedle.elbow], color='r', linestyle='--', 
                   label=f'Elbow at {distances[kneedle.elbow]:.2f}')
    plt.title(f'k-Distance Graph (k={min_samples})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return distances[kneedle.elbow] if kneedle.elbow is not None else None

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample datasets with different characteristics
    n_samples = 1000
    
    # 1. Moons dataset (non-linear separation)
    X_moons, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    
    # 2. Circles dataset (concentric circles)
    X_circles, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    
    # 3. Blobs with noise (varying density)
    X_blobs, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    X_blobs = np.vstack([X_blobs, np.random.uniform(low=-10, high=10, size=(100, 2))])
    
    # Standardize features
    scaler = StandardScaler()
    datasets = {
        'Moons': scaler.fit_transform(X_moons),
        'Circles': scaler.fit_transform(X_circles),
        'Blobs with Noise': scaler.fit_transform(X_blobs)
    }
    
    # DBSCAN parameters
    min_samples = 5
    
    plt.figure(figsize=(18, 12))
    
    for i, (name, X) in enumerate(datasets.items(), 1):
        # Find optimal eps using k-distance graph
        eps = find_optimal_eps(X, min_samples)
        if eps is None:
            eps = 0.5  # Default value if no elbow found
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        
        # Calculate silhouette score (excluding noise points)
        if len(np.unique(y_pred)) > 1 and np.any(y_pred != -1):
            mask = (y_pred != -1)
            if len(np.unique(y_pred[mask])) > 1:  # Need at least 2 clusters
                score = silhouette_score(X[mask], y_pred[mask])
            else:
                score = -1
        else:
            score = -1
        
        # Plot original data
        plt.subplot(2, 3, i)
        plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
        plt.title(f'{name} (Original Data)')
        plt.grid(True, alpha=0.3)
        
        # Plot clustering results
        plt.subplot(2, 3, i + 3)
        plot_clusters(X, y_pred, f'DBSCAN: {name}\neps={eps:.2f}, min_samples={min_samples}\nSilhouette: {score:.2f}')
    
    plt.tight_layout()
    plt.savefig('dbscan_clustering.png')
    plt.show()
    
    # Parameter sensitivity analysis
    X = datasets['Moons']
    eps_values = np.linspace(0.1, 1.0, 10)
    min_samples_values = range(2, 21, 2)
    
    plt.figure(figsize=(15, 10))
    
    for i, min_samples in enumerate([2, 5, 10], 1):
        scores = []
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(X)
            
            # Calculate silhouette score (excluding noise points)
            mask = (y_pred != -1)
            if len(np.unique(y_pred[mask])) > 1:
                score = silhouette_score(X[mask], y_pred[mask]) if np.any(mask) else -1
            else:
                score = -1
            scores.append(score)
        
        plt.plot(eps_values, scores, 'o-', label=f'min_samples={min_samples}')
    
    plt.title('DBSCAN Parameter Sensitivity Analysis')
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Silhouette Score (higher is better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dbscan_parameter_sensitivity.png')
    plt.show()
    
    # Compare with K-Means on non-linear data
    from sklearn.cluster import KMeans
    
    X = datasets['Circles']
    
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    plt.title('Original Data')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
    plt.title('K-Means Clustering')
    
    plt.subplot(1, 3, 3)
    plot_clusters(X, y_dbscan, 'DBSCAN Clustering')
    
    plt.tight_layout()
    plt.savefig('kmeans_vs_dbscan.png')
    plt.show()

if __name__ == "__main__":
    main()
