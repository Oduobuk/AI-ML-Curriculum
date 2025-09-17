"""
Exercise 2: Hierarchical Clustering

In this exercise, you'll work with hierarchical clustering and learn to
create and interpret dendrograms using different linkage methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def plot_dendrogram(model, **kwargs):
    """Create linkage matrix and plot dendrogram."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix,
        **kwargs
    )

def main():
    # Generate sample data
    np.random.seed(42)
    X, y = make_blobs(
        n_samples=200,
        centers=[[4, 4], [-2, -1], [1, 1], [10, 10]],
        cluster_std=0.9
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different linkage methods to compare
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    # Create figure for dendrograms
    plt.figure(figsize=(20, 10))
    
    for i, method in enumerate(linkage_methods, 1):
        plt.subplot(2, 2, i)
        
        # Calculate the linkage matrix
        Z = linkage(X_scaled, method=method)
        
        # Plot the dendrogram
        dendrogram(Z, p=5, truncate_mode='level')
        plt.title(f'Dendrogram - {method.capitalize()} Linkage')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('hierarchical_dendrograms.png')
    plt.show()
    
    # Let's perform clustering with the best method (Ward)
    from sklearn.cluster import AgglomerativeClustering
    
    # Fit the model
    cluster = AgglomerativeClustering(
        n_clusters=4,
        affinity='euclidean',
        linkage='ward'
    )
    cluster_labels = cluster.fit_predict(X_scaled)
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('Hierarchical Clustering Results (Ward Linkage)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True, alpha=0.3)
    plt.savefig('hierarchical_clusters.png')
    plt.show()
    
    # Let's also visualize the cophenetic correlation coefficient
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    
    print("\nCophenetic Correlation Coefficients:")
    print("-" * 40)
    for method in linkage_methods:
        Z = linkage(X_scaled, method=method)
        c, coph_dists = cophenet(Z, pdist(X_scaled))
        print(f"{method.capitalize():<10}: {c:.4f}")
        
    # Determine optimal number of clusters using the elbow method
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(idxs, last_rev, 'b-', marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_method.png')
    plt.show()

if __name__ == "__main__":
    main()
