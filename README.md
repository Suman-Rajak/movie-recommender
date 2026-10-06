# GenAI-Task20-SumanRajak

## Assignment 20 - Building and Deploying a Movie Recommendation System

A content-based movie recommender. You pick a movie, and the app shows similar movies based on their plot, tagline, genres and keywords (TF-IDF + cosine similarity).

**Live app:** ADD-YOUR-RENDER-LINK-HERE
**GitHub repo:** ADD-YOUR-GITHUB-REPO-LINK-HERE

## Dataset

**TMDB 5000 Movie Dataset** (Kaggle)
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

I used `tmdb_5000_movies.csv` (4,803 movies). The item is the movie `title`, and the text features are `overview`, `tagline`, `genres` and `keywords`.

## Files

- `app.py` - Streamlit app
- `recommendation_system.ipynb` - notebook with Tasks 1-5 (preprocessing, TF-IDF, similarity, recommend function)
- `data/tmdb_5000_movies.csv` - original Kaggle file
- `data/movies_clean.csv` - cleaned data made by the notebook and used by the app
- `requirements.txt` - Python packages for Render
- `.python-version` - Python version for Render

## What I did

1. Loaded the dataset and checked shape, columns, missing values and duplicate titles. Some titles repeat (Batman, The Host), so I show the year with each title
2. Pulled the names out of the JSON `genres` and `keywords` columns and combined them with the overview and tagline
3. Cleaned the text: lowercase, removed punctuation and stopwords (scikit-learn's English list), and filled missing values with ""
4. TF-IDF with `max_features=5000` and `ngram_range=(1, 2)`, giving a 4803 x 5000 matrix
5. Cosine similarity between all movies, and a `recommend(item_name, top_n=5)` function, tested on Avatar, The Dark Knight, Toy Story and Titanic
6. A Streamlit app with a dropdown, a slider and a Recommend button

In the app I only compute the similarity row for the selected movie instead of the full 4803 x 4803 matrix (about 180 MB), so it fits on Render's free plan.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
