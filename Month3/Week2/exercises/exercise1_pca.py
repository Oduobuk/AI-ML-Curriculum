"""
Exercise 1: PCA Implementation and Comparison

In this exercise, you'll implement PCA from scratch and compare it with scikit-learn's implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MyPCA:
    """A simple implementation of Principal Component Analysis (PCA)."""
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        # 1. Standardize data
        self.mean_ = np.mean(X, axis=0)
        X_std = X - self.mean_
        
        # 2. Compute covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Store components
        self.components_ = eigenvectors.T[:self.n_components]
        
        # 6. Calculate explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        
        return self
        
    def transform(self, X):
        X_std = X - self.mean_
        return np.dot(X_std, self.components_.T)

def main():
    # Load and preprocess data
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target
    
    # Custom PCA
    my_pca = MyPCA(n_components=2)
    X_my_pca = my_pca.fit(X).transform(X)
    
    # Scikit-learn PCA
    sklearn_pca = PCA(n_components=2)
    X_sklearn_pca = sklearn_pca.fit_transform(X)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_my_pca[:, 0], X_my_pca[:, 1], c=y, cmap='viridis')
    plt.title('Custom PCA')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_sklearn_pca[:, 0], X_sklearn_pca[:, 1], c=y, cmap='viridis')
    plt.title('Scikit-learn PCA')
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png')
    plt.show()
    
    # Compare explained variance
    print("\nCustom PCA explained variance ratio:", my_pca.explained_variance_ratio_)
    print("Scikit-learn PCA explained variance ratio:", sklearn_pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
