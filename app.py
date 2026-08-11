"""Streamlit dashboard for the AI Movie Recommender.

Three tabs, all reading the project's single sources of truth:
    • Metrics       ← experiments/metrics.json (same file the README uses)
    • Data & EDA    ← datasets/ml-latest-small via src.data
    • Recommend     ← live top-K from a fitted Recommender

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# Load OPENAI_API_KEY (and friends) from a local .env if present. No-op in
# deploys that inject real environment variables / secrets instead.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.chat import MovieChatAgent
from src.data import MovieLens, load_movielens
from src.reporting import load_metrics, render_metrics_table
from src.recommenders import PopularityRecommender
from src.recommenders.base import Recommender
from src.tools import MovieTools

METRICS_PATH = Path("experiments/metrics.json")

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")


# --- cached loaders --------------------------------------------------------

@st.cache_data(show_spinner="Loading MovieLens…")
def get_data() -> MovieLens:
    return load_movielens()


@st.cache_resource(show_spinner="Fitting model…")
def get_fitted_model() -> Recommender:
    ml = load_movielens()
    return PopularityRecommender(scoring="count").fit(ml.ratings)


@st.cache_resource(show_spinner="Preparing chat agent…")
def get_agent() -> MovieChatAgent:
    from src.semantic import SemanticSearch

    ml = load_movielens()
    model = PopularityRecommender(scoring="count").fit(ml.ratings)
    semantic = SemanticSearch.load_if_available()  # None if embeddings absent
    return MovieChatAgent(MovieTools(ml, model, semantic=semantic))


# --- tabs ------------------------------------------------------------------

def tab_metrics() -> None:
    st.subheader("📊 Evaluation metrics")
    if not METRICS_PATH.exists():
        st.info("No metrics yet. Run `python main.py` to generate "
                "`experiments/metrics.json`.")
        return

    metrics = load_metrics(METRICS_PATH)
    df = pd.DataFrame(metrics).T  # models as rows
    k = int(df["k"].iloc[0])
    metric_cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]

    display = df[metric_cols].copy()
    display.index = [name.replace("_", " ").title() for name in display.index]
    st.dataframe(display.style.format("{:.3f}"), use_container_width=True)

    st.caption("Per-metric comparison across models")
    st.bar_chart(display, use_container_width=True)

    with st.expander("Raw Markdown (as embedded in the README)"):
        st.code(render_metrics_table(metrics), language="markdown")


def tab_eda(ml: MovieLens) -> None:
    st.subheader("🔍 Dataset & EDA")
    r = ml.ratings

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings", f"{len(r):,}")
    c2.metric("Users", f"{ml.n_users:,}")
    c3.metric("Rated movies", f"{ml.n_movies:,}")
    c4.metric("Avg rating", f"{r['rating'].mean():.2f}")

    left, right = st.columns(2)

    with left:
        st.caption("Rating value distribution")
        st.bar_chart(r["rating"].value_counts().sort_index(), use_container_width=True)

        st.caption("Cold start — ratings per movie")
        per_movie = r.groupby("movieId").size()
        cold = int((per_movie <= 2).sum())
        st.write(
            f"**{cold:,}** movies ({cold / len(per_movie) * 100:.1f}%) have ≤ 2 "
            f"ratings — the long tail that popularity alone struggles with."
        )

    with right:
        st.caption("Top genres")
        genre_counts = Counter(g for gs in ml.movies["genres"] for g in gs)
        top = pd.Series(dict(genre_counts.most_common(10))).sort_values()
        st.bar_chart(top, use_container_width=True)

    st.caption("Most-rated movies")
    title = ml.movies.set_index("movieId")["title"]
    most = per_movie.sort_values(ascending=False).head(10)
    st.dataframe(
        pd.DataFrame({"title": title.reindex(most.index).values, "ratings": most.values}),
        use_container_width=True, hide_index=True,
    )


def tab_recommend(ml: MovieLens, model: Recommender) -> None:
    st.subheader("🎯 Recommendations")
    st.caption(
        "The current model is a **popularity baseline** — it recommends broadly "
        "loved movies the user hasn't seen. Personalisation arrives with "
        "collaborative filtering."
    )

    user_ids = sorted(ml.ratings["userId"].unique())
    col_a, col_b, col_c = st.columns([2, 2, 1])
    user_id = col_a.selectbox("User", user_ids)
    all_genres = sorted({g for gs in ml.movies["genres"] for g in gs})
    genre = col_b.selectbox("Filter by genre", ["(any)"] + all_genres)
    k = col_c.slider("Top-K", 5, 25, 10)

    title = ml.movies.set_index("movieId")["title"]
    genres = ml.movies.set_index("movieId")["genres"]
    seen = set(ml.ratings.loc[ml.ratings["userId"] == user_id, "movieId"])

    # This user's own favourites, for context.
    user_ratings = ml.ratings[ml.ratings["userId"] == user_id]
    faves = user_ratings.sort_values("rating", ascending=False).head(5)
    with st.expander(f"User {user_id}'s top-rated movies ({len(user_ratings)} ratings)"):
        st.dataframe(
            pd.DataFrame({
                "title": title.reindex(faves["movieId"]).values,
                "rating": faves["rating"].values,
            }),
            use_container_width=True, hide_index=True,
        )

    # Pull a generous candidate list, then optionally filter by genre.
    candidates = model.recommend(user_id, k=500, exclude=seen)
    if genre != "(any)":
        candidates = [m for m in candidates if genre in genres.get(m, [])]
    top = candidates[:k]

    st.markdown(f"**Top {len(top)} recommendations for user {user_id}**")
    st.dataframe(
        pd.DataFrame({
            "rank": range(1, len(top) + 1),
            "title": [title.get(m, m) for m in top],
            "genres": [", ".join(genres.get(m, [])) for m in top],
        }),
        use_container_width=True, hide_index=True,
    )


def tab_chat() -> None:
    st.subheader("💬 Chat")
    st.caption(
        "Ask in plain language — e.g. *\"I like Harry Potter, what should I "
        "watch?\"*. The LLM calls the recommender tools and can only suggest "
        "movies those tools return (no hallucinated titles)."
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.info(
            "Chat needs an `OPENAI_API_KEY`. Locally, copy `.env.example` to "
            "`.env` and put your key there:\n\n"
            "```bash\ncp .env.example .env   # then edit OPENAI_API_KEY\n```\n"
            "In a deploy, inject it as an environment variable / secret instead."
        )
        return

    agent = get_agent()
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("What do you feel like watching?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = agent.reply(st.session_state.chat)
                except Exception as exc:  # keep the UI alive on API errors
                    answer = f"⚠️ {exc}"
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})


# --- main ------------------------------------------------------------------

st.title("🎬 AI Movie Recommender")

ml = get_data()
model = get_fitted_model()

metrics_tab, eda_tab, rec_tab, chat_tab = st.tabs(
    ["📊 Metrics", "🔍 Data & EDA", "🎯 Recommend", "💬 Chat"]
)
with metrics_tab:
    tab_metrics()
with eda_tab:
    tab_eda(ml)
with rec_tab:
    tab_recommend(ml, model)
with chat_tab:
    tab_chat()
