import os
import pickle
from textblob import TextBlob
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dropout, Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPool1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords

_dir = os.path.join(os.path.dirname(__file__), 'pickles')

def process_text(line, p_array, n_array):
    word_array = line.split(" ")
    p_count = 0
    n_count = 0
    for i in range(len(word_array)):
        blob = TextBlob(word_array[i])
        p = round(blob.sentiment.polarity, 5)
        if p >= 0.5:
            p_count += 1
        else:
            n_count += 1
    p_array.append(p_count)
    n_array.append(n_count)

def train_and_save_cnn():
    dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
    p_array, n_array, s_array, po_array = [], [], [], []
    MAX_WORDS, EMBED_DIM = 10000, 120
    clean_data = []
    for i in range(len(dataset)):
        words = [w for w in dataset['headline']
                [i].split() if not w in stopwords.words('english')]
        clean_data.append(words)

    for i in range(len(dataset)):
        process_text(dataset['headline'][i], p_array, n_array)
        blob = TextBlob(dataset['headline'][i])
        s_array.append(round(blob.sentiment.subjectivity, 5))
        po_array.append(round(blob.sentiment.polarity, 5) + 1)


    tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=True)
    tokenizer.fit_on_texts(dataset['headline'])
    text = tokenizer.texts_to_sequences(dataset['headline'])
    sequences = pad_sequences(text, maxlen=120)
    p = pd.DataFrame(np.column_stack((sequences, p_array)))
    p = pd.DataFrame(np.column_stack((p, n_array)))
    p = pd.DataFrame(np.column_stack((p, s_array)))
    p = pd.DataFrame(np.column_stack((p, po_array)))

    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, EMBED_DIM))
    model.add(Conv1D(EMBED_DIM, 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(EMBED_DIM, 2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(GlobalMaxPool1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(
        p, dataset['is_sarcastic'], test_size=0.25, random_state=464)

    history = model.fit(X_train, y_train, verbose=1, epochs=15, batch_size=32,
                        validation_data=(X_test, y_test))

    with open(os.path.join(_dir, 'sarcasm_tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    model.save(os.path.join(_dir, 'sarcasm_cnn_model.h5'))

def process_line(line):
    p_array, n_array = [], []
    word_array = line.split(" ")
    p_count, n_count = 0, 0
    for i in range(len(word_array)):
        blob = TextBlob(word_array[i])
        p = round(blob.sentiment.polarity, 5)
        if p >= 0.5 : 
            p_count += 1
        else:
            n_count += 1
    p_array.append(p_count)
    n_array.append(n_count)
    return p_array, n_array
    
def _process_text(text):
    clean_data, p_array, n_array, s_array, po_array = [], [], [], [], []
    words = [w for w in text.split() if not w in stopwords.words('english')]
    clean_data.append("".join(words))
    p_array, n_array = process_line(text)
    blob = TextBlob(text)
    s_array.append(round(blob.sentiment.subjectivity, 5))
    po_array.append(round(blob.sentiment.polarity, 5) + 1)
    return p_array, n_array, s_array, po_array

def predict_sarcasm(text):
    with open(os.path.join(_dir, 'sarcasm_tokenizer.pkl'), 'rb') as f:
        sarcasm_tokenizer = pickle.load(f)
    cnn_sarcasm = load_model(os.path.join(_dir, 'sarcasm_cnn_model.h5'))
    tokenize_string = sarcasm_tokenizer.texts_to_sequences([text])
    string_sequence = pad_sequences(tokenize_string, maxlen = 120)
    p_array, n_array, s_array, po_array = _process_text("" + text)
    p = pd.DataFrame(np.column_stack((string_sequence, p_array)))
    p = pd.DataFrame(np.column_stack((p, n_array)))
    p = pd.DataFrame(np.column_stack((p, s_array)))
    p = pd.DataFrame(np.column_stack((p, po_array)))
    return cnn_sarcasm.predict(p)[0][0] * 100
