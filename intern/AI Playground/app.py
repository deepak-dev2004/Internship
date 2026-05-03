import streamlit as st
from models import generate_text, caption_image, answer_question
from utils import load_image

st.set_page_config(page_title="Multi-Modal AI Playground")

st.title("🤖 Multi-Modal AI Playground")

option = st.sidebar.selectbox(
    "Choose Mode",
    ["Text Generation", "Image Captioning", "Question Answering"]
)

# ---------------- TEXT ----------------
if option == "Text Generation":
    st.header("📝 Text Generation")

    prompt = st.text_area("Enter prompt")

    if st.button("Generate"):
        result = generate_text(prompt)
        st.write(result)

# ---------------- IMAGE ----------------
elif option == "Image Captioning":
    st.header("🖼️ Image Captioning")

    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("Generate Caption"):
            caption = caption_image(image)
            st.success(caption)

# ---------------- QA ----------------
elif option == "Question Answering":
    st.header("❓ Question Answering")

    context = st.text_area("Enter context")
    question = st.text_input("Ask a question")

    if st.button("Get Answer"):
        answer = answer_question(context, question)
        st.success(answer)