from __future__ import annotations

from src.data.data_loader import MovieDataLoader
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_recommender import ContentRecommender
from src.models.collaborative_recommender import CollaborativeRecommender


class RecommendationService:
    def __init__(self, data_dir: str = "data") -> None:
        loader = MovieDataLoader(data_dir=data_dir)
        self.movies = loader.load_movies()
        self.ratings = loader.load_ratings()

        self.popularity = PopularityRecommender()
        self.popularity.fit(self.movies)

        self.content = ContentRecommender()
        self.content.fit(self.movies)

        self.collaborative = CollaborativeRecommender()
        self.collaborative.fit(self.movies, self.ratings)

    def get_popularity_recommendations(self, top_n: int, min_votes: int | None = None):
        return self.popularity.recommend(top_n=top_n, min_votes=min_votes)

    def get_content_recommendations(self, movie_title: str, top_n: int):
        return self.content.recommend(movie_title=movie_title, top_n=top_n)

    def get_collaborative_recommendations(self, user_id: int, top_n: int):
        return self.collaborative.recommend(user_id=user_id, top_n=top_n)
