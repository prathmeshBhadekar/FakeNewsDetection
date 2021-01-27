#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from textblob import TextBlob
from profanity_check import predict, predict_prob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.externals import joblib
#from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
#import re
#import matplotlib.pyplot as plt
from sklearn import metrics

app = Flask(__name__)

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

def FeatureSet(frame):
    global_set = []
    for row in frame:
        text = row
        profanity = predict_prob([text])[0]
        sentiment = TextBlob(text)
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        # return [polarity, subjectivity, profanity]
        global_set.append([polarity, subjectivity, profanity])
    return global_set


#def SarcasticSet(frame):
#    global_set = []
#    for row in frame:
#        text = row
#        profanity = predict_prob([text])[0]
#        sentiment = TextBlob(text)
#        polarity = sentiment.polarity
#        # return [polarity, subjectivity, profanity]
#        global_set.append([polarity, profanity])
#    return global_set
#
#
#def sarcasticFeature(text):
#    profanity = predict_prob([text])[0]
#    sentiment = TextBlob(text)
#    polarity = sentiment.polarity
#    return [profanity, polarity]


#with open('tfidf_title.pkl', 'rb') as f:
#    tfidf_title = pickle.load(f)
#
#with open('passive_title.pkl', 'rb') as f:
#    passive_title = pickle.load(f)
#
#with open('svc_title.pkl', 'rb') as f:
#    svc_title = pickle.load(f)
#
#
#def calltext(text, tfidf_title, passive_title):
#    # text = "hello this text is about autheticating news"
#
#    title = np.array(FeatureSet([text]))
#
#    transform = tfidf_title.transform([text])
#
#    tfidf = transform.todense()
#
#    pred = np.column_stack((tfidf, title))
#
#    predict_text = passive_title.predict(pred)
#    return predict_text
#
#
#with open('tfidf_main.pkl', 'rb') as f:
#    tfidf_main = pickle.load(f)
#
#with open('passive.pkl', 'rb') as f:
#    passive = pickle.load(f)
#
#
#def calltitle(text, tfidf_main, passive):
#    # text = " This text is fake "
#    transform_main = tfidf_main.transform([text])
#
#    tfidf_main = transform_main.todense()
#
#    title_main = np.array(FeatureSet([text]))
#
#    pred_main = np.column_stack((tfidf_main, title_main))
#
#    predict_title = passive.predict(pred_main)
#
#    return predict_title


with open('Main/MainRnn-28-04.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Main/MainTokenizer-28-04.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def maincontent(text,tokenizer, model):
    text = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(text,maxlen=100,padding='post')
    pred = 0    
    with graph.as_default():
    	pred = model.predict(sequences)[0][0]
    return np.round(pred)



with open('Bias/tfidfBias-28-04.pkl', 'rb') as f:
    tfidf_bias = pickle.load(f)

with open('Bias/passiveBias-28-04.pkl', 'rb') as f:
    passive_bias = pickle.load(f)

#with open('svc_bias.pkl', 'rb') as f:
#    svc_bias = pickle.load(f)


def callbias(bias_text, bias_tfidf, passive_bias):
    bias_transform = tfidf_bias.transform([bias_text])
    bias_output = passive_bias.predict(bias_transform)
    return bias_output


#with open('Sarcasm/tokenizerCNN-28-04.pkl', 'rb') as f:
#    tfidf_sarcasm = pickle.load(f)
#
#with open('Sarcasm/sarcasmCNN-28-04.pkl', 'rb') as f:
#    passive_sarcasm = pickle.load(f)
#
#
#def sarcasm(tfidf_sarcasm, passive_sarcasm, text):
#    # text = "hello this text is about autheticating news"
#
#    title = np.array(SarcasticSet([text]))
#
#    transform = tfidf_sarcasm.transform([text])
#
#    tfidf = transform.todense()
#
#    pred = np.column_stack((tfidf, title))
#
#    output = passive_sarcasm.predict(pred)
#    return output

import pickle

with open('Sarcasm/tokenizerCNN-28-04.pkl', 'rb') as f:
    tokenizer_sarcasm = pickle.load(f)   
        
with open('Sarcasm/sarcasmCNN-28-04.pkl', 'rb') as f:
    cnn_sarcasm = pickle.load(f)   

#text = "Mirrors can't talk, lucky for you they can't laugh either."


def Fun( line ):
    p_array = []
    n_array = []
    word_array = line.split(" ")
    p_count=0
    n_count=0
    for i in range(len(word_array)):
        blob = TextBlob(word_array[i])
        p = round(blob.sentiment.polarity, 5)
        if p>=0.5 : 
            p_count+=1
        else:
            n_count+=1
    p_array.append(p_count)
    n_array.append(n_count)
    return p_array,n_array


    
def Processing(text):
    clean_data = []    
    p_array=[]
    n_array=[]   
    s_array=[]
    po_array=[]  
    words = [w for w in text.split() if not w in stopwords.words('english')]
    clean_data.append("".join(words))
    p_array,n_array = Fun(text)
    blob=TextBlob(text)
    s_array.append(round(blob.sentiment.subjectivity, 5))
    po_array.append(round(blob.sentiment.polarity, 5)+1)
    return p_array,n_array,s_array,po_array

#API
def Sarcasm(text, tokenizer_sarcasm, cnn_sarcasm):
    tokenize_string = tokenizer_sarcasm.texts_to_sequences([text])
    string_sequence = pad_sequences(tokenize_string, maxlen=100)
    p_array,n_array,s_array,po_array = Processing(""+text)
    p=pd.DataFrame(np.column_stack((string_sequence, p_array)))
    p=pd.DataFrame(np.column_stack((p,n_array)))
    p=pd.DataFrame(np.column_stack((p,s_array)))
    p=pd.DataFrame(np.column_stack((p,po_array)))
    pred = 0    
    with graph.as_default():
    	pred = model.predict(string_sequence)[0][0]
    return np.round(pred)
    
#return model.predict(p)[0][0]*100

#call as
#print(Sarcasm(text, tokenizer_sarcasm, cnn_sarcasm))    



def predictoutput(text, title):
    a = maincontent(title, tokenizer, model)
    b = maincontent(text, tokenizer, model)
    c = Sarcasm(text, tokenizer_sarcasm, cnn_sarcasm)
    d = callbias(text, tfidf_bias, passive_bias)
    return [a, b, c, d[0]]


@app.route('/', methods=['GET', 'POST'])
def res():
    if request.method == 'POST':
        title = request.form.get('headline')
        text = request.form.get('comment')

        result = predictoutput(text, title)

        print(result)

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction='')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # text = request.form.to_dict(['comment'])
    # title = request.form.to_dict(['headline'])
    # x_text = list(x_text.values())
    # x_text = list(map(int, x_text))
    # x_title = list(x_title.values())
    # x_title = list(map(int, x_title))
    # x_sarcasm = list(x_sarcasm.values())
    # x_sarcasm = list(map(int, x_sarcasm))
    # text = list(text.values())
    # text = list(map(int, text))
    # title = list(title.values())
    # title = list(map(int, title))
    # result = predictoutput(text,title)

    # if result == [1,1,1,0]:
    #     prediction='Title: Fake Text: Fake, Sarcasm : True'
    # elif result == [0,0,0,0]  :
    #     prediction = 'Title: Real Text: Real, Sarcasm: False'

    # if result[0] == 1:
    #     prediction = 'Given text is fake'
    # else:
    #     prediction = 'Given text is real'

    # if result[1] == 1:
    #     prediction = 'Given headline is fake'
    # else:
    #     prediction = 'Given headline is real'

    # if result[2] == 1:
    #     prediction = 'Given content is sarcastic'
    # else:
    #     prediction = 'Given content is not sarcastic'

    # return render_template("predict.html", prediction=prediction)


