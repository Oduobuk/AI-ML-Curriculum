"""
<<<<<<< HEAD
Exercise 1: Principal Component Analysis (PCA) Implementation

In this exercise, you'll implement PCA from scratch using NumPy and apply it to various datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

class PCA:
    """
    Principal Component Analysis (PCA) implementation from scratch.
    """
    def __init__(self, n_components=2):
        """
        Initialize PCA with the number of components to keep.
        
        Parameters:
        n_components (int): Number of principal components to keep
        """
=======
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
>>>>>>> student-branch
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
<<<<<<< HEAD
        """
        Fit the PCA model to the data.
        
        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features)
        """
        # 1. Standardize the data
=======
        # 1. Standardize data
>>>>>>> student-branch
        self.mean_ = np.mean(X, axis=0)
        X_std = X - self.mean_
        
        # 2. Compute covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
<<<<<<< HEAD
        # 4. Sort eigenvalues and eigenvectors in descending order
=======
        # 4. Sort in descending order
>>>>>>> student-branch
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
<<<<<<< HEAD
        # 5. Store the first n_components eigenvectors as principal components
        self.components_ = eigenvectors[:, :self.n_components]
        
        # 6. Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features)
        
        Returns:
        numpy.ndarray: Transformed data, shape (n_samples, n_components)
        """
        X_std = X - self.mean_
        return np.dot(X_std, self.components_)
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features)
        
        Returns:
        numpy.ndarray: Transformed data, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

def plot_explained_variance(pca, title="Explained Variance Ratio"):
    """Plot the explained variance ratio for each principal component."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, 
            alpha=0.7, 
            align='center',
            label='Individual explained variance')
    
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 
             where='mid',
             label='Cumulative explained variance')
    
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_with_sklearn(X, n_components=2):
    """Compare our implementation with scikit-learn's PCA."""
    # Our implementation
    our_pca = PCA(n_components=n_components)
    X_our_pca = our_pca.fit_transform(X)
    
    # Scikit-learn's implementation
    sklearn_pca = SklearnPCA(n_components=n_components, random_state=42)
    X_sklearn_pca = sklearn_pca.fit_transform(X)
    
    # Compare the results
    print("Our PCA explained variance ratio:", our_pca.explained_variance_ratio_)
    print("Sklearn PCA explained variance ratio:", sklearn_pca.explained_variance_ratio_)
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_our_pca[:, 0], X_our_pca[:, 1], alpha=0.7)
    plt.title('Our PCA Implementation')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_sklearn_pca[:, 0], X_sklearn_pca[:, 1], alpha=0.7)
    plt.title("Scikit-learn's PCA")
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Apply our PCA implementation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Plot the explained variance ratio
    pca_full = PCA(n_components=X_std.shape[1])
    pca_full.fit(X_std)
    plot_explained_variance(pca_full, "Explained Variance Ratio (Iris Dataset)")
    
    # Visualize the first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Iris Dataset')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    
    # Compare with scikit-learn's implementation
    compare_with_sklearn(X_std)
    
    # Apply to a more complex dataset (Digits)
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # Standardize the data
    X_digits_std = scaler.fit_transform(X_digits)
    
    # Apply PCA
    pca_digits = PCA(n_components=2)
    X_digits_pca = pca_digits.fit_transform(X_digits_std)
    
    # Visualize the digits in 2D
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], 
                         c=y_digits, cmap='tab10', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Digits Dataset')
    plt.colorbar(scatter, label='Digit')
    plt.grid(True, alpha=0.3)
    
    plt.show()
=======
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
>>>>>>> student-branch

if __name__ == "__main__":
    main()
