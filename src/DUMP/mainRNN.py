
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution2D, Embedding
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten, LSTM,GRU,Bidirectional
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from profanity_check import predict, predict_prob
import matplotlib.pyplot as plt 

X = pd.read_csv('train.csv')
X = X.dropna(subset = ['text'])

x = X['text']
y = X['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=8)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X['text'])

max_pad = max([len(s.split()) for s in X['text']])
max_pad = 100

#max_pad = 250
vocabulary = len(tokenizer.word_index)+1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

prat = "Hello its me i was wondering if this news is fake"
trans = tokenizer.texts_to_sequences([prat])


X_train_pad = pad_sequences(X_train,maxlen= max_pad, padding = 'post')
X_test_pad = pad_sequences(X_test,maxlen= max_pad, padding = 'post')


EMBED_DIMENSION = 50

model = Sequential()

model.add(Embedding(vocabulary, EMBED_DIMENSION,input_length = max_pad))

model.add(GRU(units=32, dropout=0.2,recurrent_dropout=0.2))

model.add(Dropout(0.2))

model.add(Dense(units=1, activation ='sigmoid'))


model.summary()



model.compile(metrics=['accuracy'],optimizer='adam',loss='binary_crossentropy')

history = model.fit(X_train_pad, y_train ,verbose = 1, epochs = 15,
          validation_data=(X_test_pad, y_test))


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()



X['label'].value_counts().plot(kind="bar", rot=0)


with open('MainRnn-28-04.pkl', 'wb') as f:
    pickle.dump(model,f)

with open('MainTokenizer-28-04.pkl', 'wb') as f:
    pickle.dump(tokenizer,f)





#API

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









