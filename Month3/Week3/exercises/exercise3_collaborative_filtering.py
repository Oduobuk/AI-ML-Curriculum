"""
Exercise 3: Collaborative Filtering
"""
import numpy as np
from scipy.sparse.linalg import svds

from data.datasets import get_ratings_data, get_movie_data

def normalize_ratings(ratings):
    """Normalize ratings by subtracting mean rating for each user."""
    user_ratings_mean = np.mean(ratings, axis=1)
    return ratings - user_ratings_mean.reshape(-1, 1), user_ratings_mean

def predict_ratings(ratings, k=2):
    """Predict ratings using SVD-based collaborative filtering."""
    # Normalize ratings
    norm_ratings, user_ratings_mean = normalize_ratings(ratings)
    
    # Perform SVD
    U, sigma, Vt = svds(norm_ratings, k=k)
    sigma = np.diag(sigma)
    
    # Reconstruct the matrix
    pred_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    
    return np.clip(pred_ratings, 1, 5)  # Clip to rating range

def recommend_items(user_id, pred_ratings, original_ratings, top_n=2):
    """Recommend items not yet rated by the user."""
    # Get user's predicted ratings
    user_pred = pred_ratings[user_id]
    
    # Find items not rated by the user
    unrated_items = np.where(original_ratings[user_id] == 0)[0]
    
    # Get top N predicted ratings for unrated items
    top_items = unrated_items[np.argsort(-user_pred[unrated_items])][:top_n]
    
    return top_items, user_pred[top_items]

def main(use_real_data=False):
    """Run the collaborative filtering example.
    
    Args:
        use_real_data: If True, uses MovieLens dataset
    """
    # Load data
    ratings_df = get_ratings_data(use_real_data=use_real_data)
    movies_df = get_movie_info(use_real_data=use_real_data)
    
    # Create user-item matrix
    # For real data, we need to map user and item IDs to matrix indices
    user_ids = ratings_df['user_id'].unique()
    item_ids = ratings_df['item_id'].unique()
    
    # Create mappings
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    # Initialize ratings matrix with zeros
    ratings = np.zeros((n_users, n_items))
    
    # Fill in the ratings
    for _, row in ratings_df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['item_id']]
        ratings[user_idx, item_idx] = row['rating']
    
    print("Original Ratings (first 5 users):")
    print(ratings[:5])
    
    # Predict ratings
    pred_ratings = predict_ratings(ratings)
    
    # Get recommendations for the first user
    user_id = 0
    item_indices, scores = recommend_items(user_id, pred_ratings, ratings)
    
    # Create reverse mapping from index to item ID
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    
    print(f"\nRecommended movies for user {user_id + 1}:")
    for idx, score in zip(item_indices, scores):
        movie_id = idx_to_item[idx]
        try:
            movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
            print(f"- {movie_title}: Predicted rating {score:.2f}")
        except IndexError:
            print(f"- Item {movie_id}: Predicted rating {score:.2f}")

if __name__ == "__main__":
    # Set to True to use real MovieLens data
    main(use_real_data=True)
