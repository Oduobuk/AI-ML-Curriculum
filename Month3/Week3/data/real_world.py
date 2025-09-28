"""
Real-world datasets for Week 3 exercises.
"""
import os
import pandas as pd
from urllib.request import urlretrieve
import zipfile

def download_movielens():
    """Download and extract MovieLens 100K dataset."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, 'ml-100k.zip')
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K dataset...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    return os.path.join(data_dir, 'ml-100k')

def load_movielens_ratings():
    """Load MovieLens 100K ratings data."""
    data_dir = download_movielens()
    ratings_path = os.path.join(data_dir, 'u.data')
    return pd.read_csv(
        ratings_path,
        sep='\t',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

def load_movielens_movies():
    """Load MovieLens 100K movie information."""
    data_dir = download_movielens()
    movies_path = os.path.join(data_dir, 'u.item')
    
    columns = ['movie_id', 'title', 'release_date', 'video_release_date',
              'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    
    return pd.read_csv(
        movies_path,
        sep='|',
        encoding='latin-1',
        header=None,
        names=columns
    )

def get_real_ratings():
    """Get real-world ratings data."""
    ratings = load_movielens_ratings()
    return ratings[['user_id', 'movie_id', 'rating']]

def get_real_movies():
    """Get real-world movie data."""
    movies = load_movielens_movies()
    return movies[['movie_id', 'title']]
