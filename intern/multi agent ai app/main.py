import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# Download tokenizer (first time only)
nltk.download('punkt')

# ---------------- AGENTS ---------------- #

def research_agent(text):
    sentences = sent_tokenize(text)

    # Word frequency
    words = text.lower().split()
    freq = Counter(words)

    # Score sentences
    scored = []
    for sent in sentences:
        score = sum(freq[word] for word in sent.lower().split() if word in freq)
        scored.append((score, sent))

    # Get top 3 sentences
    top_sentences = sorted(scored, reverse=True)[:3]

    research = [sent for score, sent in top_sentences]
    return research


def writer_agent(research_points):
    article = "Introduction:\n"
    article += "This topic discusses important aspects.\n\n"

    article += "Main Points:\n"
    for point in research_points:
        article += f"- {point}\n"

    article += "\nConclusion:\n"
    article += "In conclusion, this topic is significant and impactful."

    return article


def reviewer_agent(article):
    # Simple improvements
    article = article.replace("  ", " ")
    article = article.strip()

    # Capitalize properly
    article = article.capitalize()

    return article


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Multi-Agent AI ", layout="centered")

st.title("🤖 Multi-Agent AI App ")


user_input = st.text_area("Enter your topic or paragraph:")

if st.button("Run Multi-Agent System"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        st.subheader("🔍 Research Agent Output")
        research = research_agent(user_input)
        for r in research:
            st.write("-", r)

        st.subheader("✍️ Writer Agent Output")
        article = writer_agent(research)
        st.write(article)

        st.subheader("✅ Reviewer Agent Output")
        final = reviewer_agent(article)
        st.success(final)