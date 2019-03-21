#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:17:11 2019

@author: pratham
"""

from random import randint
from telegram.ext import CommandHandler, Updater, Dispatcher, MessageHandler, Filters
import telegram
import pandas as pd
import re
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

TOKEN = 'TOKEN'


def greet(bot, update):
    greetings = ['bonjour', 'hola', 'hallo', 'sveiki', 'namaste', 'szia', 'halo', 'ciao']
    rint = randint(0, len(greetings) - 1)
    update.message.reply_text(greetings[rint]+ " !")


def paladin(bot, update):
    string_1 = "Hello! I am Paladin, a Bot made to talk with you.\n"
    string_2 = "You can try using the commands \n "
    string_3 = "/greet :  Paladin greets you \n"
    string_4 = "/paladin : To know about paladin\n"
    string_5 = "/sentiment : To find the sentiment of a sentence\n"
    string_6 = "/nn : To know news fake or real using a pre-trained neural network"
    update.message.reply_text(string_1 + string_2 + string_3 + string_4 + string_5 + string_6)


def sentiment(bot, update):
    text = update.message.text
    text = text[10:]
    review = Query(text)+""
    update.message.reply_text(review)


def noCommand(bot, update):
    string = update.message.text
    #user_name = update.message.user.first_name
    #print(user_name)
    update.message.reply_text(user_name + "I have no idea what "+string+" means ")



def Query(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    #print(review)
    review = review.lower()
    #print(review)
    review = review.split()
    #print(review)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    print(review)
    return review



def nn(bot, update):
    text = update.message.text
    text = text[4:]
    reply = "" + str(transform(text))
    update.message.reply_text(reply)         



    
def transform(review):
    
    with open('clf_1.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    with open('tfidf_vector.pkl', 'rb') as f:
        vector = pickle.load(f)

    a = [review]
    pred = vector.transform(a)
    x = clf.predict(pred)
    print(x)
    return x
    


def main():
    updater = Updater(token = TOKEN)
    dispatcher = updater.dispatcher

    handler = CommandHandler('greet', greet)
    dispatcher.add_handler(handler)

    handler = CommandHandler('paladin', paladin)
    dispatcher.add_handler(handler)

    handler = CommandHandler('sentiment', sentiment)
    dispatcher.add_handler(handler)
    
    message_handler = MessageHandler(Filters.text, noCommand)
    dispatcher.add_handler(message_handler)
    
    handler  = CommandHandler('nn', nn)
    dispatcher.add_handler(handler)



    updater.start_polling()


if __name__ == '__main__':
    main()
