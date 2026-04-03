from __future__ import annotations

from pathlib import Path
import pandas as pd


class MovieDataLoader:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_movies(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "movies.csv")

    def load_ratings(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "ratings.csv")

    def load_credits(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "credits.csv")
