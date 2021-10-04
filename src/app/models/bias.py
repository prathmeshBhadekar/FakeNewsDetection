import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier

_dir = os.path.join(os.path.dirname(__file__), 'pickles')

def train_and_save_bias():
    frame = pd.read_csv(os.path.join(_dir, 'bias_training_with_col.csv'))
    frame = frame.set_index("title")
    y = frame.bias
    frame.drop("bias", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(frame['text'], y, test_size=0.4, random_state=20)
    tfidf_vector = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range = (1, 3)) 
    tfidf_train, tfidf_test = tfidf_vector.fit_transform(X_train), tfidf_vector.transform(X_test)

    #Passive Aggresive Classifier
    classifier = PassiveAggressiveClassifier(max_iter=30)
    classifier.fit(tfidf_train, y_train)
    pred = classifier.predict(tfidf_test)
    score = accuracy_score(y_test, pred)
    print("accuracy: %s" % (score))

    with open(os.path.join(_dir, 'bias_classifier.pkl'), 'wb') as s:
        pickle.dump(classifier, s)

    with open(os.path.join(_dir, 'bias_tfidf_vector.pkl'), 'wb') as s:
        pickle.dump(tfidf_vector, s)


def get_bias(text):
    with open(os.path.join(_dir, 'bias_tfidf_vector.pkl'), 'rb') as f:
        tfidf_bias = pickle.load(f)

    with open(os.path.join(_dir, 'bias_classifier.pkl'), 'rb') as f:
        passive_bias = pickle.load(f)

    bias_transform = tfidf_bias.transform([text])
    bias_output = passive_bias.predict(bias_transform)
    bias_ = {
            0: "Left Bias",
            1: "Neutral Bias",
            2: "Right Bias"
    }
    return bias_[bias_output[0]]

