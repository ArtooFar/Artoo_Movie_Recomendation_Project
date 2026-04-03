import pandas as pd

from surprise import Dataset, Reader, SVD



class CollaborativeRecommender:
    def __init__(self) -> None:
        self.model = None
        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.ratings = ratings[["userId", "movieId", "rating"]].copy()


        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(self.ratings, reader)
        trainset = dataset.build_full_trainset()
        self.model = SVD()
        self.model.fit(trainset)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if self.movies is None or self.ratings is None:
            raise ValueError("CollaborativeRecommender must be fitted before calling recommend().")

        watched = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])

        candidates = self.movies[["id", "title", "release_date", "overview"]].copy()

        candidates["movieId"] = pd.to_numeric(candidates["id"], errors="coerce")
        candidates = candidates.dropna(subset=["movieId"]).copy()
        candidates["movieId"] = candidates["movieId"].astype(int)

        candidates = candidates.loc[~candidates["movieId"].isin(watched)].copy()

        candidates["predicted_rating"] = candidates["movieId"].apply(lambda movie_id: self.model.predict(user_id, int(movie_id)).est)


        return candidates[["title", "release_date", "predicted_rating", "overview"]].sort_values(
            "predicted_rating", ascending=False
        ).head(top_n)
