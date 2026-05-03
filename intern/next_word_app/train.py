import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ---------- LOAD DATA ----------
with open("corpus.txt", "r") as f:
    data = f.read().lower()

# ---------- TOKENIZATION ----------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

total_words = len(tokenizer.word_index) + 1

# ---------- CREATE SEQUENCES ----------
input_sequences = []

for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# ---------- PADDING ----------
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# ---------- SPLIT ----------
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# ---------- ONE HOT ----------
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=total_words)

# ---------- MODEL ----------
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_seq_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------- TRAIN ----------
model.fit(X, y, epochs=100, verbose=1)

# ---------- SAVE ----------
model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save max sequence length
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_seq_len, f)