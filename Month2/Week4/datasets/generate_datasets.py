"""
Script to generate sample datasets for Week 4 exercises.
"""
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles
import pandas as pd

def generate_imbalanced_data():
    """Generate an imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        flip_y=0.05,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), y

def generate_nonlinear_data():
    """Generate non-linear dataset for kernel methods."""
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    return pd.DataFrame(X, columns=['feature_1', 'feature_2']), y

def save_datasets():
    """Generate and save all datasets."""
    # Imbalanced dataset
    X_imb, y_imb = generate_imbalanced_data()
    X_imb['target'] = y_imb
    X_imb.to_csv('imbalanced_data.csv', index=False)
    
    # Non-linear dataset
    X_nl, y_nl = generate_nonlinear_data()
    X_nl['target'] = y_nl
    X_nl.to_csv('nonlinear_data.csv', index=False)
    
    print("Datasets generated successfully!")

if __name__ == "__main__":
    save_datasets()
