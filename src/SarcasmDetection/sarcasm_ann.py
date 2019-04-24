#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:56:31 2019

@author: aniket
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob

# Importing the dataset
dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines =True)
dataset=dataset.iloc[0:26000,1:]

data=[]
for i in range(26000):
    blob=TextBlob(dataset['headline'][i]) 
    p=round(blob.sentiment.polarity, 5)
    s=round(blob.sentiment.subjectivity, 5)
    data.append([ dataset['headline'][i], p, s, dataset['is_sarcastic']])    
    
dataframe = pd.DataFrame(data, columns=['headline','polarity','subjectivity','is_sarcastic'])



# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 26000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['headline'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#X_text = cv.fit_transform(text).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 100))

classifier.add(Dropout(0.2))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 50,activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)