import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load dataset
df = pd.read_csv("books.csv")

# Combine features
df['content'] = df['author'] + " " + df['genre'] + " " + df['description']

# Vectorization
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(df['content'])

# Similarity
similarity = cosine_similarity(matrix)

# Function to get paperback link (Open Library API)
def get_paperback_link(book_title):
    url = f"https://openlibrary.org/search.json?title={book_title}"
    try:
        res = requests.get(url).json()
        if res['docs']:
            key = res['docs'][0].get('key', '')
            return f"https://openlibrary.org{key}"
    except:
        return None

# Recommendation function
def recommend(book):
    if book not in df['title'].values:
        return []

    idx = df[df['title'] == book].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    rec_books = []
    for i in scores:
        title = df.iloc[i[0]]['title']
        link = get_paperback_link(title)
        rec_books.append((title, link))
    
    return rec_books

# Streamlit UI
st.title("📚 Book Recommendation Engine")

selected_book = st.selectbox("Choose a book", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_book)

    st.subheader("Recommended Books:")

    for book, link in recommendations:
        st.write(f"### {book}")
        if link:
            st.markdown(f"[📖 View Paperback]({link})")
        else:
            st.write("Paperback link not available")