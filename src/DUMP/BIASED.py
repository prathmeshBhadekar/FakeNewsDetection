#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Apr  8 10:30:48 2019
@author: pratham
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier

#0:neutral 1:Left 2:Right

#Reading data as pandas dataframe
frame = pd.read_csv('bias_training_with_col.csv')

#Inspecing Shape
frame.shape

#Inspecting top 5 rows
frame.head()

#Setting the DataFrame index (row labels) using one or more existing columns
frame = frame.set_index("title")
frame.head()

y = frame.bias
y.head()

frame.drop("bias", axis=1)
frame.head()

X_train, X_test, y_train, y_test = train_test_split(frame['text'], y, test_size=0.05, random_state=53)

X_train.head()

y_train.head()


# Initialize the tfidf_vectorizer 
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range = (1, 3)) 

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)

#print(tfidf_test)
X_test.shape

## Get the feature names of tfidf_vectorizer 
#print(tfidf_vectorizer.get_feature_names()[-10:])


#Passive Aggresive Classifier 
pa_tfidf_clf = PassiveAggressiveClassifier(n_iter=50)
pa_tfidf_clf.fit(tfidf_train, y_train)
pred = pa_tfidf_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


#Linear Support Vector Classifier
svc_tfidf_clf = LinearSVC()
svc_tfidf_clf.fit(tfidf_train, y_train)
pred = svc_tfidf_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


import pickle

with open('passive_bias.pkl', 'wb') as s:
        pickle.dump(pa_tfidf_clf, s)

with open('svc_bias.pkl', 'wb') as s:
        pickle.dump(svc_tfidf_clf, s)

with open('tfidf_bias.pkl', 'wb') as s:
        pickle.dump(tfidf_vectorizer, s)





