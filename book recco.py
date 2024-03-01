import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class BookRecommendationEngine:
    def __init__(self, k=5):
        self.k = k
        self.model = None
        self.books = None
        self.book_indices = None

    def fit(self, book_features):
        # Convert book features to sparse matrix
        self.books = csr_matrix(book_features)
        
        # Fit kNN model
        self.model = NearestNeighbors(n_neighbors=self.k, algorithm='brute', metric='cosine')
        self.model.fit(self.books)

        # Store book indices for later use
        self.book_indices = {book: idx for idx, book in enumerate(book_features.index)}

    def recommend_books(self, book_id, num_recommendations=5):
        # Find k nearest neighbors
        distances, indices = self.model.kneighbors(self.books[self.book_indices[book_id]], n_neighbors=num_recommendations+1)
        
        # Exclude the input book itself
        distances = distances.squeeze()[1:]
        indices = indices.squeeze()[1:]

        # Retrieve book IDs of recommended books
        recommended_books = [list(self.book_indices.keys())[list(self.book_indices.values()).index(idx)] for idx in indices]

        return recommended_books, distances

# Example usage
if __name__ == "__main__":
    # Sample book features
    book_features = {
        'book_1': [1, 0, 1, 1, 0],
        'book_2': [0, 1, 1, 0, 1],
        'book_3': [1, 1, 0, 1, 0],
        'book_4': [0, 1, 0, 1, 1],
        'book_5': [1, 0, 1, 0, 1],
    }

    # Convert to DataFrame for better representation
    import pandas as pd
    book_features_df = pd.DataFrame.from_dict(book_features, orient='index', columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'])

    # Initialize recommendation engine
    engine = BookRecommendationEngine(k=2)
    engine.fit(book_features_df)

    # Get recommendations
    book_id = 'book_1'
    recommended_books, distances = engine.recommend_books(book_id)
    print(f"Recommended books for {book_id}: {recommended_books} with distances {distances}")
