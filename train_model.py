import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import re
import pickle

# Load data
input_file = 'holmes.txt'
with open(input_file, 'r', encoding='utf-8') as infile:
    data = infile.read()

# Limit data for faster training in this environment
data = data[:100000]

def remove_emojis_and_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(' +', ' ', text)
    return text

def preprocess_pipeline(data):
    sentences = data.split('\n')
    processed_sentences = []
    for s in sentences:
        s = remove_emojis_and_special_characters(s).strip()
        if len(s) > 0:
            processed_sentences.append(s.lower())
    return processed_sentences

tokenized_sentences = preprocess_pipeline(data)

# Tokenizer
tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(tokenized_sentences)
total_words = len(tokenizer.word_index) + 1

# Save Tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Sequences
input_sequences = []
for line in tokenized_sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Feature and Label
X, labels = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Model
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    Bidirectional(LSTM(100)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train (few epochs for demo purpose)
print("Starting training...")
model.fit(X, y, epochs=10, verbose=1)

# Save Model
model.save('sentence_completion_model.h5')
print("Model and Tokenizer saved!")

# Save max_sequence_len
with open('config.pkl', 'wb') as f:
    pickle.dump({'max_sequence_len': max_sequence_len}, f)
