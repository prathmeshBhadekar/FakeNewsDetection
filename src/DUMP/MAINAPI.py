#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 02:35:27 2019

@author: aniket
"""

import pickle
from keras.preprocessing.text import Tokenizer
from keras.layers import GRU,Dense,LSTM
from keras.preprocessing.sequence import pad_sequences



# MAIN API

with open('MainRnn-28-04.pkl', 'rb') as f:
    model = pickle.load(f)

with open('MainTokenizer-28-04.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def maincontent(text,tokenizer, model):
    text = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(text,maxlen=100,padding='post')
    pred = model.predict(sequences)[0][0]
    return pred*100

text = "Hello its me "
print(maincontent(text,tokenizer,model))

