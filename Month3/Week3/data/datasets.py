"""
Dataset module for Week 3 exercises.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from .real_world import get_real_ratings, get_real_movies

# Sample movie data for recommendation exercises
MOVIES = {
    'movie_id': range(1, 11),
    'title': [
        'The Dark Knight', 'Inception', 'The Shawshank Redemption',
        'The Godfather', 'Pulp Fiction', 'Interstellar',
        'The Matrix', 'Fight Club', 'Forrest Gump', 'The Prestige'
    ],
    'genre': [
        'Action Crime Drama', 'Action Adventure Sci-Fi', 'Drama',
        'Crime Drama', 'Crime Drama', 'Adventure Drama Sci-Fi',
        'Action Sci-Fi', 'Drama', 'Drama Romance', 'Drama Mystery'
    ],
    'director': [
        'Christopher Nolan', 'Christopher Nolan', 'Frank Darabont',
        'Francis Ford Coppola', 'Quentin Tarantino', 'Christopher Nolan',
        'Lana Wachowski', 'David Fincher', 'Robert Zemeckis', 'Christopher Nolan'
    ]
}

def get_movie_data():
    """Return movie data as a pandas DataFrame."""
    return pd.DataFrame(MOVIES)

def generate_anomaly_data(n_samples=1000, contamination=0.05, random_state=42):
    """
    Generate synthetic dataset for anomaly detection.
    
    Returns:
        X: Feature matrix (n_samples, 2)
        y: Labels (1 for inliers, -1 for outliers)
    """
    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers
    
    # Generate inliers (normal data)
    X_inliers, _ = make_blobs(
        n_samples=n_inliers,
        centers=[[0, 0], [5, 5]],
        cluster_std=0.5,
        random_state=random_state
    )
    
    # Generate outliers
    np.random.seed(random_state)
    X_outliers = np.random.uniform(low=-5, high=10, size=(n_outliers, 2))
    
    # Combine and create labels
    X = np.vstack([X_inliers, X_outliers])
    y = np.ones(n_samples, dtype=int)
    y[-n_outliers:] = -1  # -1 for outliers
    
    return X, y

def get_ratings_data(use_real_data=False):
    """Return user-movie ratings.
    
    Args:
        use_real_data: If True, uses MovieLens 100K dataset
        
    Returns:
        DataFrame with columns ['user_id', 'movie_id', 'rating']
    """
    if use_real_data:
        return get_real_ratings()
    else:
        # Sample synthetic data
        ratings = [
            (1, 1, 5), (1, 2, 4), (1, 5, 3),
            (2, 2, 5), (2, 3, 4), (2, 6, 5),
            (3, 1, 4), (3, 4, 5), (3, 7, 4),
            (4, 3, 5), (4, 5, 4), (4, 8, 3),
            (5, 2, 4), (5, 6, 5), (5, 9, 4)
        ]
        return pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'])

def get_movie_info(use_real_data=False):
    """Return movie information.
    
    Args:
        use_real_data: If True, uses MovieLens movie information
        
    Returns:
        DataFrame with movie information
    """
    if use_real_data:
        return get_real_movies()
    else:
        return get_movie_data()
