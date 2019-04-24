#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:21:56 2019

@author: aniket
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines =True)
dataset=dataset.iloc[0:26700,1:]

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
    
data=[]
for i in range(len(dataset)):
    fun(dataset['headline'][i])
    
dataset_1 = pd.DataFrame(np.column_stack(( dataset , p_array)))
dataset_2 = pd.DataFrame(np.column_stack(( dataset_1 , n_array)))
    