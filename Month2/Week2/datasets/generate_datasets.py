"""
Script to generate sample datasets for Week 2 exercises on Logistic Regression and KNN.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_linear_separable_data(n_samples=300, random_state=42):
    """Generate a linearly separable 2D dataset for binary classification."""
    np.random.seed(random_state)
    
    # Generate positive and negative samples
    X1 = np.random.normal(loc=2, scale=1, size=(n_samples//2, 2))
    X2 = np.random.normal(loc=5, scale=1, size=(n_samples//2, 2))
    X = np.vstack([X1, X2])
    
    # Create labels (0 for first class, 1 for second class)
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    # Add some noise
    noise = np.random.normal(0, 0.5, size=X.shape)
    X += noise
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    
    return df


def generate_moons_data(n_samples=300, noise=0.2, random_state=42):
    """Generate a non-linearly separable 2D dataset (two interleaving half circles)."""
    from sklearn.datasets import make_moons
    
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Scale the features
    X = X * 5  # Scale for better visualization
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    
    return df


def generate_imbalanced_data(n_samples=1000, weights=[0.9, 0.1], random_state=42):
    """Generate an imbalanced binary classification dataset."""
    n_classes = len(weights)
    n_samples_per_class = [int(n_samples * w) for w in weights]
    
    # Generate features
    X_list = []
    y_list = []
    
    for class_idx, n in enumerate(n_samples_per_class):
        # Different means for each class
        mean = [class_idx * 3, class_idx * 3]
        cov = [[1, 0.5], [0.5, 1]]  # Some correlation between features
        
        X_class = np.random.multivariate_normal(mean, cov, n)
        y_class = np.ones(n, dtype=int) * class_idx
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    # Combine all classes
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle the data
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target'] = y
    
    return df


def generate_high_dimensional_data(n_samples=500, n_features=20, n_informative=10, 
                                 n_redundant=5, n_classes=3, random_state=42):
    """Generate a high-dimensional dataset for multi-class classification."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Create DataFrame
    feature_columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['target'] = y
    
    return df


def save_datasets():
    """Generate and save all datasets."""
    # Linear separable data
    linear_data = generate_linear_separable_data()
    linear_data.to_csv('datasets/linear_separable.csv', index=False)
    
    # Moons data (non-linear)
    moons_data = generate_moons_data()
    moons_data.to_csv('datasets/moons_data.csv', index=False)
    
    # Imbalanced data
    imbalanced_data = generate_imbalanced_data()
    imbalanced_data.to_csv('datasets/imbalanced_data.csv', index=False)
    
    # High-dimensional data
    high_dim_data = generate_high_dimensional_data()
    high_dim_data.to_csv('datasets/high_dimensional.csv', index=False)
    
    # Split one dataset into train/test for the exercises
    X = linear_data.drop('target', axis=1)
    y = linear_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Save train/test splits
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv('datasets/train_data.csv', index=False)
    
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv('datasets/test_data.csv', index=False)
    
    print("All datasets have been generated and saved to the 'datasets' directory.")


if __name__ == "__main__":
    import os
    
    # Create datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Generate and save all datasets
    save_datasets()
