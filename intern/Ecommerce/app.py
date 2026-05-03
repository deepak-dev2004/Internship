import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE ----------
st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("🛒 E-commerce Customer Behavior & Recommendation System")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce.csv")
    df['combined'] = df['category'] + " " + df['description']
    return df

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("User Selection")
user_id = st.sidebar.selectbox("Select User ID", df['user_id'].unique())

# ---------- USER DATA ----------
user_data = df[df['user_id'] == user_id]

st.subheader(f"👤 Customer {user_id} Behavior")

col1, col2, col3 = st.columns(3)

col1.metric("Total Purchases", len(user_data))
col2.metric("Total Spend (₹)", int(user_data['price'].sum()))
col3.metric("Avg Purchase (₹)", int(user_data['price'].mean()))

st.write("### Purchased Products")
st.dataframe(user_data[['product', 'category', 'price']])

# ---------- TOP CATEGORY ----------
top_category = user_data['category'].mode()[0]
st.write(f"🔥 Most Purchased Category: **{top_category}**")

# ---------- VECTORIZE ----------
@st.cache_data
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(data['combined'])
    return cosine_similarity(matrix)

similarity = compute_similarity(df)

# ---------- RECOMMEND ----------
def recommend_for_user(user_id):
    user_items = df[df['user_id'] == user_id].index.tolist()

    scores = {}
    for idx in user_items:
        sim_scores = list(enumerate(similarity[idx]))
        for i, score in sim_scores:
            if i not in user_items:
                scores[i] = scores.get(i, 0) + score

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    recommended = [df.iloc[i[0]]['product'] for i in sorted_items[:5]]
    return recommended

# ---------- BUTTON ----------
if st.button("🎯 Recommend Products"):
    recs = recommend_for_user(user_id)

    st.subheader("Recommended Products")
    for item in recs:
        st.write("👉", item)