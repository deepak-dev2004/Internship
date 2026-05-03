import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- LOAD ----------
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_seq_len = pickle.load(f)

# ---------- PREDICTION ----------
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    predicted = np.argmax(model.predict(token_list), axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return "Not found"

# ---------- UI ----------
st.title("🔮 LSTM Next Word Prediction")

input_text = st.text_input("Enter a sentence:")

if input_text:
    next_word = predict_next_word(input_text.lower())
    st.subheader("Next Word Prediction:")
    st.success(next_word)