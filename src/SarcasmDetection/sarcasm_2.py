#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:28:30 2019

@author: aniket
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
"""
blob = TextBlob("Analytics Vidhya is a great platform to learn data science. \n It helps community through blogs, hackathons, discussions,etc.")
blob.sentiment
"""

# Importing the dataset
dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines =True)
dataset=dataset.iloc[0:26000,1:]

data=[]
for i in range(26000):
    blob=TextBlob(dataset['headline'][i]) 
    p=round(blob.sentiment.polarity, 5)
    s=round(blob.sentiment.subjectivity, 5)
    data.append([ dataset['headline'][i], p, s, dataset['is_sarcastic'][i]])    
    
dataframe = pd.DataFrame(data, columns=['headline','polarity','subjectivity','is_sarcastic'])
#Creating a dataset for user input
"""d = {'headline' : ["When I was a boy, I had a disease that required me to eat dirt three times a day in order to survive... It's a good thing my older brother told me about it.","Smoking will kill you... Bacon will kill you... But,smoking bacon will cure it."], 'is_sarcastic' : [1,1]}
dataframe = pd.DataFrame(data=d)
"""
text=["hbvhfv"]
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
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
