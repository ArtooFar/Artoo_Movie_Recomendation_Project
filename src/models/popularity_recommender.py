from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    def __init__(self, quantile: float = 0.90) -> None:
        self.quantile = quantile
        self.movies: pd.DataFrame | None = None
        self.minimum_votes: float | None = None
        self.global_mean: float | None = None

    def fit(self, movies: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.minimum_votes = self.movies["vote_count"].quantile(self.quantile)
        self.global_mean = self.movies["vote_average"].mean()

    def recommend(self, top_n: int = 10, min_votes: int | None = None) -> pd.DataFrame:
        if self.movies is None or self.minimum_votes is None or self.global_mean is None:
            raise ValueError("PopularityRecommender must be fitted before calling recommend().")

        threshold = float(min_votes) if min_votes is not None else float(self.minimum_votes)
        filtered = self.movies.loc[self.movies["vote_count"] >= threshold].copy()
        filtered["weighted_rating"] = (
            (filtered["vote_count"] / (filtered["vote_count"] + threshold)) * filtered["vote_average"]
            + (threshold / (threshold + filtered["vote_count"])) * self.global_mean
        )

        columns = ["title", "release_date", "vote_average", "vote_count", "weighted_rating", "overview"]
        return filtered[columns].sort_values("weighted_rating", ascending=False).head(top_n)
