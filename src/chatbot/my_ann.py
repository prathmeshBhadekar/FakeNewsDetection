#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 28 14:35:44 2019

@author: pratham
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
 
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from textblob import TextBlob

from profanity_check import predict, predict_prob


""" DONE WITH ALL IMPORTS """


le = LabelEncoder()

frame = pd.read_csv('fake_or_real_news.csv')

"""
EXTRACT PROFANITY, SUBJECTIVITY, POLARITY
"""

def Extract(text):
    profanity = predict_prob([text])[0]
    sentiment = TextBlob(text)
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    return [polarity, subjectivity, profanity]
    

#x = Extract(frame['text'].iloc[0])
    

def FeatureSet(frame):
    global_set = []
    for row in frame :
        text = row
        profanity = predict_prob([text])[0]
        sentiment = TextBlob(text)
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        #return [polarity, subjectivity, profanity]
        global_set.append([polarity, subjectivity, profanity])
    return global_set
        


news_features = FeatureSet(frame['text'].iloc[0:6000])

"""SKIP THIS LINE """
news_features = pd.DataFrame(news_features, columns =
                             ['polarity', 'subjectivity', 'profanity'])

""" skipped """



news_features = np.array(news_features)
add_label = frame['label'].iloc[0:6000]

train_label = le.fit_transform(add_label)


X_train, X_test, y_train, y_test = train_test_split(news_features, train_label, test_size=0.35, random_state=44)


y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

"""X_train = np.array(X_train)
nsamples, nx, ny = X_train.shape
X_train_dataset = X_train.reshape((nsamples,nx*ny))
"""


"""

clf = Sequential()

clf.add(Dense(256, input_dim = 3, activation = 'relu'))

clf.add(Dropout(0.2))
clf.add(Dense(256, activation = 'relu'))

clf.add(Dropout(0.2))
clf.add(Dense(1, activation = 'sigmoid'))


clf.summary()

clf.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


clf.fit(X_train, y_train, batch_size = 32, epochs = 100)


prediction = clf.predict(X_test)

prediction = np.argmax(np.round(prediction), axis=1)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, prediction)

"""

#make models
clf = MultinomialNB() 
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


#Passive Aggresive Classifier 
pa_tfidf_clf = PassiveAggressiveClassifier(n_iter=50)
pa_tfidf_clf.fit(X_train, y_train)
pred = pa_tfidf_clf.predict(X_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


#Linear Support Vector Classifier
svc_tfidf_clf = LinearSVC()
svc_tfidf_clf.fit(X_train, y_train)
pred = svc_tfidf_clf.predict(X_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)






