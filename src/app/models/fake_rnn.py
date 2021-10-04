import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GRU, Conv2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

_dir = os.path.join(os.path.dirname(__file__), 'pickles')

def train_and_save_rnn():
    """Train and Save the Fake/Real RNN"""
    X = pd.read_csv('train.csv')
    EMBED_DIMENSION = 10
    X = X.dropna(subset=['text'])
    x, y = X['text'], X['label']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=6)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X['text'])
    max_pad = max([len(s.split()) for s in X['text']])
    max_pad = 50

    vocabulary = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train, maxlen=max_pad, padding='post')
    X_test_pad = pad_sequences(X_test, maxlen=max_pad, padding='post')

    model = Sequential()
    model.add(Embedding(vocabulary, EMBED_DIMENSION, input_length=max_pad))
    model.add(GRU(units=32, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dropout(0.26))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()
    model.compile(metrics=['accuracy'], optimizer='adam',
                  loss='binary_crossentropy')
    history = model.fit(X_train_pad, y_train, verbose=1, epochs=50,
                        validation_data=(X_test_pad, y_test))
    # Save models
    model.save(os.path.join(_dir, 'main_model.h5'))
    with open(os.path.join(_dir, 'mrnn_tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)


def predict_fake(text: str):
    """API to predict Fake/Real"""
    text = str(text)
    model = load_model(os.path.join(_dir, 'main_model.h5'))
    with open(os.path.join(_dir, 'mrnn_tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)

    text = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(text, maxlen=100, padding='post')
    pred = model.predict(sequences)[0][0]
    return pred * 100


