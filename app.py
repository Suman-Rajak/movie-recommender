"""
Movie Recommender - content-based recommendation system (Streamlit app)
Dataset: TMDB 5000 Movie Dataset (Kaggle)
Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


@st.cache_data
def load_movies():
    # cleaned data created by recommendation_system.ipynb
    movies = pd.read_csv("data/movies_clean.csv")
    movies["clean_text"] = movies["clean_text"].fillna("")
    movies["overview"] = movies["overview"].fillna("")
    movies["genre_names"] = movies["genre_names"].fillna("")
    return movies


@st.cache_resource
def build_tfidf(texts):
    # same settings as in the notebook
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    return tfidf.fit_transform(texts)


movies = load_movies()
tfidf_matrix = build_tfidf(movies["clean_text"].tolist())
index_of = pd.Series(movies.index, index=movies["display_title"])


def recommend(item_name, top_n=5):
    """Return the top_n movies most similar to item_name."""
    idx = index_of[item_name]
    # similarity of the selected movie with every movie (one row only, saves memory)
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked = [(i, s) for i, s in ranked if i != idx][:top_n]
    result = movies.iloc[[i for i, _ in ranked]].copy()
    result["similarity"] = [s for _, s in ranked]
    return result


# ---------------- UI ----------------
st.title("🎬 Movie Recommender")
st.write("Pick a movie you like and get similar movies, based on the plot, genres and keywords.")

titles = sorted(movies["display_title"].tolist())
default_index = titles.index("The Dark Knight (2008)") if "The Dark Knight (2008)" in titles else 0

selected = st.selectbox("Choose a movie", titles, index=default_index)
top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Recommend"):
    chosen = movies.iloc[index_of[selected]]
    st.caption(f"Selected: {chosen['display_title']} | {chosen['genre_names']}")
    st.subheader(f"Movies similar to {selected}")

    for rank, (_, row) in enumerate(recommend(selected, top_n).iterrows(), start=1):
        st.markdown(f"**{rank}. {row['display_title']}**")
        st.write(f"Genres: {row['genre_names']}  |  Rating: {row['vote_average']}/10  |  "
                 f"Similarity: {row['similarity']:.2f}")
        overview = row["overview"]
        st.caption(overview[:250] + ("..." if len(overview) > 250 else ""))

st.divider()
st.caption("Data: TMDB 5000 Movie Dataset (Kaggle). Content-based filtering with TF-IDF + cosine similarity.")
