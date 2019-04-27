#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:16:58 2019
@author: pratham
"""

from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU, Embedding
from keras.models import Sequential
from keras.layers import GlobalMaxPool1D, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import pandas as pd
import numpy as np
import re




dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines =True)

clean_data = []
for i in range(len(dataset)):
    words = re.sub('[^a-zA-Z]', ' ', dataset['headline'] [i])
    words = words.lower()
    words = words.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    words = ' '.join(words)
    clean_data.append(words)
    
p_array=[]
n_array=[]   
def fun( line ):
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
    
s_array=[]
po_array=[]    

for i in range(len(dataset)):
    fun(dataset['headline'][i])
    blob=TextBlob(dataset['headline'][i])
    s_array.append(round(blob.sentiment.subjectivity, 5))
    po_array.append(round(blob.sentiment.polarity, 5)+1)
    
max_words = 10000

tokenizer  = Tokenizer(num_words = max_words, char_level = True)
tokenizer.fit_on_texts(dataset['headline'])
text = tokenizer.texts_to_sequences(dataset['headline'])

sequences = pad_sequences(text, maxlen = 120)
p=pd.DataFrame(np.column_stack((sequences,p_array)))
p=pd.DataFrame(np.column_stack((p,n_array)))
p=pd.DataFrame(np.column_stack((p,s_array)))
p=pd.DataFrame(np.column_stack((p,po_array)))


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

X_train, X_test, y_train, y_test = train_test_split(p, dataset['is_sarcastic'], test_size = 0.25, random_state = 464)

history = model.fit(X_train,y_train, verbose = 1, epochs = 50 , batch_size = 32 ,
          validation_data = (X_test, y_test))


import matplotlib.pyplot as plt
#%matplotlib qt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss CNN SARCASM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import pickle    
with open('pk_sarcasm_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)   

import pickle    
with open('sarcasm_denseCNN.pkl', 'wb') as f:
    pickle.dump(model, f)   
