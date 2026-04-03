from __future__ import annotations

from pathlib import Path
import streamlit as st

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.recommendation_service import RecommendationService


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource

def load_service() -> RecommendationService:
    data_dir = ROOT / "data"
    return RecommendationService(data_dir=str(data_dir))


service = load_service()

st.title("🎬 Movie Recommendation System")

mode = st.sidebar.radio(
    "Recommendation mode",
    ["Popularity-Based", "Content-Based", "Collaborative Filtering"],
)

top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)


def render_movie_cards(df, score_column: str) -> None:
    for _, row in df.iterrows():
        title = row.get("title", "Unknown title")
        subtitle = row.get("release_date", "")
        score = row.get(score_column)
        overview = row.get("overview", "") or "No overview available."
        with st.container(border=True):
            st.subheader(title)
            cols = st.columns([1, 3])
            with cols[0]:
                if score is not None:
                    label = score_column.replace("_", " ").title()
                    if isinstance(score, float):
                        st.metric(label, f"{score:.3f}" if score_column != "vote_average" else f"{score:.1f}")
                    else:
                        st.metric(label, score)
                if subtitle:
                    st.caption(subtitle)
            with cols[1]:
                st.write(overview)


if mode == "Popularity-Based":
    st.subheader("Top ranked movies by weighted rating")
    min_votes = st.sidebar.number_input("Minimum votes override", min_value=0, value=0, step=100)
    recommendations = service.get_popularity_recommendations(top_n=top_n, min_votes=min_votes or None)
    st.dataframe(
        recommendations[["title", "vote_average", "vote_count", "weighted_rating"]],
        width='stretch',
        hide_index=True,
    )
    render_movie_cards(recommendations, "weighted_rating")

elif mode == "Content-Based":
    st.subheader("Find movies similar to a selected title")
    movie_title = st.selectbox("Choose a movie", sorted(service.movies["title"].dropna().unique().tolist()), index=0)
    recommendations = service.get_content_recommendations(movie_title=movie_title, top_n=top_n)
    st.dataframe(
        recommendations[["title", "vote_average", "similarity"]],
        width='stretch',
        hide_index=True,
    )
    render_movie_cards(recommendations, "similarity")

else:
    st.subheader("Personalized recommendations with SVD")
    max_user_id = int(service.ratings["userId"].max())
    user_id = st.sidebar.number_input("User ID", min_value=1, max_value=max_user_id, value=15, step=1)
    try:
        recommendations = service.get_collaborative_recommendations(user_id=user_id, top_n=top_n)
        st.dataframe(
            recommendations[["title", "predicted_rating"]],
            width='stretch',
            hide_index=True,
        )
        render_movie_cards(recommendations, "predicted_rating")
    except ImportError as exc:
        st.warning(str(exc))
        st.info("This page matches the original notebook approach and depends on scikit-surprise.")
        st.code("pip install -r requirements.txt")
