#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from newsapi import NewsApiClient
#import requests  

user_input=str(input())

api = NewsApiClient(api_key='873176fa76a04365990955ee05a0832c')

head_lines = api.get_top_headlines(q=user_input , country='in')


datastore = api.get_everything(q=user_input,language='en',sort_by='relevancy')
#print(head_lines)

print("\n\n HeadLines \n")

i=0
while i < head_lines['totalResults'] :
    print("\n" + "Title : \n" + head_lines['articles'][i]['title'])
    print("\n" + "Description : \n" + head_lines['articles'][i]['description'])
    if datastore['articles'][i]['content'] != None :
        print("\n" + "Content : \n" + head_lines['articles'][i]['content'])
    i+=1

print("\n\n Other News \n")


i=0
while i<20 :
    print("\n" + "Title : \n" + datastore['articles'][i]['title'])
    print("\n" + "Description : \n" + datastore['articles'][i]['description'])
    if datastore['articles'][i]['content'] != None :
        print("\n" + "Content : \n" + datastore['articles'][i]['content'])
    i+=1





    
