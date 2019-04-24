#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:54:00 2019

@author: aniket
"""

from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU, Embedding
from keras.models import Sequential
from keras.layers import GlobalMaxPool1D, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np


dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines =True)

"""
dataset=dataset

data=[]
for i in range(26000):
    blob=TextBlob(dataset['headline'][i]) 
    p=round(blob.sentiment.polarity, 5)
    s=round(blob.sentiment.subjectivity, 5)
    data.append([ dataset['headline'][i], p, s, dataset['is_sarcastic'][i]])    
    
dataframe = pd.DataFrame(data, columns=['headline','polarity','subjectivity','is_sarcastic'])
"""


max_words = 10000

tokenizer  = Tokenizer(num_words = max_words, char_level = True)
tokenizer.fit_on_texts(dataset['headline'])
text = tokenizer.texts_to_sequences(dataset['headline'])

sequences = pad_sequences(text, maxlen = 120)
#p=pd.DataFrame(np.column_stack((sequences,dataset['headline'])))
embed_dim = 120

model = Sequential()

model.add(Embedding(len(tokenizer.word_index)+1, embed_dim))

model.add(Conv1D(embed_dim,2, activation = 'relu'))
model.add(Dropout(0.2))

#model.add(LSTM(100))
model.add(Conv1D(embed_dim,2, activation = 'relu'))
model.add(Dropout(0.3))

model.add(GlobalMaxPool1D())
model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

model.summary()

X_train, X_test, y_train, y_test = train_test_split(sequences, dataset['is_sarcastic'], test_size = 0.25, random_state = 464)

model.fit(X_train,y_train, verbose = 1, epochs = 15 , batch_size = 32 ,
          validation_data = (X_test, y_test))



#text=""





















