"""
Exercise 2: Content-Based Recommendation System
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data.datasets import get_movie_data

def get_recommendations(title, df, top_n=2):
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'] + ' ' + df['director'])
    
    # Compute similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get recommendations
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return df['title'].iloc[[i[0] for i in sim_scores]].tolist()

def main(use_real_data=False):
    """Run the content-based recommendation example.
    
    Args:
        use_real_data: If True, uses MovieLens dataset
    """
    # Get movie data
    movies_df = get_movie_info(use_real_data=use_real_data)
    
    if use_real_data:
        # Use a movie from the MovieLens dataset
        example_movie = movies_df['title'].iloc[0]
        print(f"\nRecommendations for '{example_movie}':")
        print(get_recommendations(example_movie, movies_df))
    else:
        # Use our synthetic example
        print(f"\nRecommendations for 'The Dark Knight':")
        print(get_recommendations('The Dark Knight', movies_df))

if __name__ == "__main__":
    # Set to True to use real MovieLens data
    main(use_real_data=True)
