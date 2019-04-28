
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from textblob import TextBlob
from profanity_check import predict, predict_prob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt
from sklearn import metrics




"""
Created on Mon Apr  8 11:29:45 2019

@author: pratham
"""


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



def SarcasticSet(frame):
    global_set = []
    for row in frame :
        text = row
        profanity = predict_prob([text])[0]
        sentiment = TextBlob(text)
        polarity = sentiment.polarity
        #return [polarity, subjectivity, profanity]
        global_set.append([polarity, profanity])
    return global_set



def sarcasticFeature(text):
    profanity = predict_prob([text])[0]
    sentiment = TextBlob(text)
    polarity = sentiment.polarity
    return [profanity, polarity]

"""

TEXT

"""


with open('tfidf_title.pkl', 'rb') as f:
    tfidf_title = pickle.load(f)

with open('passive_title.pkl', 'rb') as f:
    passive_title = pickle.load(f)

with open('svc_title.pkl', 'rb') as f:
    svc_title = pickle.load(f)


def calltext(text, tfidf_title, passive_title):
    #text = "hello this text is about autheticating news"
    
    title = np.array(FeatureSet([text]))
    
    transform = tfidf_title.transform([text])
    
    tfidf = transform.todense()
    
    pred = np.column_stack((tfidf, title))
    
    output = passive_title.predict(pred)
    return output

"""
TITLE
"""

with open('tfidf_main.pkl', 'rb') as f:
    tfidf_main = pickle.load(f)

with open('passive.pkl', 'rb') as f:
    passive = pickle.load(f)


def calltitle(text, tfidf_main, passive):
    #text = " This text is fake "   
    transform_main = tfidf_main.transform([text])
    
    tfidf_main = transform_main.todense()
    
    title_main = np.array(FeatureSet([text]))
    
    pred_main = np.column_stack((tfidf_main, title_main))
    
    output_main = passive.predict(pred_main)
    
    return output_main


"""
BIAS

"""

with open('tfidf_bias.pkl', 'rb') as f:
    tfidf_bias = pickle.load(f)

with open('passive_bias.pkl', 'rb') as f:
    passive_bias = pickle.load(f)

with open('svc_bias.pkl', 'rb') as f:
    svc_bias = pickle.load(f)


def callbias(bias_text, bias_tfidf, passive_bias):
    bias_transform = tfidf_bias.transform([bias_text])
    bias_output = passive_bias.predict(bias_transform)
    return bias_output



with open('tfidf_sarcasm.pkl', 'rb') as f:
   tfidf_sarcasm = pickle.load(f)


with open('passive_sarcasm.pkl', 'rb') as f:
   passive_sarcasm = pickle.load(f)



def sarcasm(tfidf_sarcasm, passive_sarcasm, text):
    #text = "hello this text is about autheticating news"
    
    title = np.array(SarcasticSet([text]))
    
    transform = tfidf_sarcasm.transform([text])
    
    tfidf = transform.todense()
    
    pred = np.column_stack((tfidf, title))
    
    output = passive_sarcasm.predict(pred)
    return output



def predictoutput(text, title):
    x = calltext(text, tfidf_main, passive)
    y = calltitle(title, tfidf_title, passive_title)
    z = sarcasm(tfidf_sarcasm, passive_sarcasm, text)
    w = callbias(text, tfidf_bias, passive_bias)
    return [x[0], y[0], z[0], w[0]]
#text title bias


text = "This news is about a boy who had to eat his own hands to survive"
title = "trump wins the presidential elections"

# TRAIN  = pd.read_csv('train.csv')

# TRAIN_1 = TRAIN.iloc[0:10]

res = predictoutput(text, title)
print(res)

