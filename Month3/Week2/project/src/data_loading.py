"""
Data loading utilities for the Fashion MNIST dataset.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_fashion_mnist():
    """
    Load the Fashion MNIST dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    
    # Convert to numpy arrays and normalize
    X = X.astype('float32') / 255.0
    y = y.astype('int')
    
    # Split into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def get_class_names():
    """Return the class names for Fashion MNIST."""
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
