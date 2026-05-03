import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df['combined_features'] = df['genre'] + " " + df['description']
    return df

df = load_data()

# ---------- VECTORIZE ----------
@st.cache_data
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(data['combined_features'])
    sim = cosine_similarity(matrix)
    return sim

similarity = compute_similarity(df)

# ---------- RECOMMEND FUNCTION ----------
def recommend(movie_title):
    if movie_title not in df['title'].values:
        return []

    index = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_scores[1:6]:
        recommendations.append(df.iloc[i[0]]['title'])

    return recommendations

# ---------- UI ----------
movie_list = df['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)

    if results:
        st.subheader("Top Recommendations:")
        for movie in results:
            st.write("👉", movie)
    else:
        st.error("Movie not found!")
