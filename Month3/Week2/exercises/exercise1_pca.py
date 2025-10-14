"""
Exercise 1: PCA Implementation and Comparison

In this exercise, you'll implement PCA from scratch and compare it with scikit-learn's implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

class MyPCA:
    """A simple implementation of Principal Component Analysis (PCA)."""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """Fit the PCA model to the data."""
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
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction to X."""
        X_std = X - self.mean_
        return np.dot(X_std, self.components_.T)
    
    def fit_transform(self, X):
        """Fit the model and apply dimensionality reduction."""
        return self.fit(X).transform(X)

def plot_explained_variance(pca, title="Explained Variance Ratio"):
    """Plot the explained variance ratio."""
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_,
            alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_),
             where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def compare_with_sklearn(X, n_components=2):
    """Compare our implementation with scikit-learn's PCA."""
    # Our implementation
    our_pca = MyPCA(n_components=n_components)
    X_our = our_pca.fit_transform(X)
    
    # Scikit-learn's implementation
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X)
    
    # Compare results
    print("Our PCA components:\n", our_pca.components_)
    print("\nScikit-learn PCA components:\n", sklearn_pca.components_)
    print("\nOur explained variance ratio:", our_pca.explained_variance_ratio_)
    print("Scikit-learn explained variance ratio:", sklearn_pca.explained_variance_ratio_)
    
    return our_pca, sklearn_pca

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Apply our PCA
    pca = MyPCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Plot explained variance
    plot_explained_variance(pca)
    
    # Compare with scikit-learn
    our_pca, sklearn_pca = compare_with_sklearn(X_std)
    
    # Visualize the results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.5, label=iris.target_names[0])
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.5, label=iris.target_names[1])
    plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], alpha=0.5, label=iris.target_names[2])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Our PCA Implementation')
    plt.legend()
    
    # Scikit-learn's PCA
    X_sklearn = sklearn_pca.transform(X_std)
    plt.subplot(1, 2, 2)
    plt.scatter(X_sklearn[y == 0, 0], X_sklearn[y == 0, 1], alpha=0.5, label=iris.target_names[0])
    plt.scatter(X_sklearn[y == 1, 0], X_sklearn[y == 1, 1], alpha=0.5, label=iris.target_names[1])
    plt.scatter(X_sklearn[y == 2, 0], X_sklearn[y == 2, 1], alpha=0.5, label=iris.target_names[2])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Scikit-learn PCA')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()