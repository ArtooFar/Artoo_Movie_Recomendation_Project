from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentRecommender:
    def __init__(self) -> None:
        self.movies: pd.DataFrame | None = None
        self.indices: pd.Series | None = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, movies: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.movies["overview"] = self.movies["overview"].fillna("")
        tfidf_matrix = self.vectorizer.fit_transform(self.movies["overview"])

        self.similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if self.movies is None or self.indices is None or self.similarity_matrix is None:
            raise ValueError("ContentRecommender must be fitted before calling recommend().")
        if movie_title not in self.indices:
            raise KeyError(f"Movie '{movie_title}' was not found in the dataset.")

        idx = int(self.indices[movie_title])
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda item: item[1], reverse=True)[1 : top_n + 1]
        movie_indices = [item[0] for item in sim_scores]

        result = self.movies.iloc[movie_indices][["title", "release_date", "vote_average", "overview"]].copy()
        result["similarity"] = [score for _, score in sim_scores]
        return result[["title", "release_date", "vote_average", "similarity", "overview"]]
