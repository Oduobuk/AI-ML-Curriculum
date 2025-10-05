"""
Preprocessing utilities for the Fashion MNIST dataset.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(n_components=None):
    """
    Create a preprocessing pipeline with scaling and optional PCA.
    
    Args:
        n_components: Number of components for PCA. If None, no PCA is applied.
        
    Returns:
        Pipeline: A scikit-learn pipeline with preprocessing steps.
    """
    steps = [
        ('scaler', StandardScaler())
    ]
    
    if n_components is not None:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
        
    return Pipeline(steps)

def apply_preprocessing(pipeline, X_train, X_test):
    """
    Apply preprocessing to training and test data.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        X_train: Training data
        X_test: Test data
        
    Returns:
        tuple: (X_train_transformed, X_test_transformed)
    """
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    return X_train_transformed, X_test_transformed
