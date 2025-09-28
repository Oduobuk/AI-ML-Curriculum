"""
Exercise 2: t-SNE and UMAP Comparison

Compare t-SNE and UMAP for visualizing high-dimensional data.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

def main():
    # Load and preprocess data
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target
    
    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Run UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Spectral', s=5)
    plt.title('t-SNE Visualization')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=5)
    plt.title('UMAP Visualization')
    
    plt.tight_layout()
    plt.savefig('tsne_umap_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
