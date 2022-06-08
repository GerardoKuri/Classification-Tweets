# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:08:49 2021

@author: gerar
"""

# -*- coding: utf-8 -*-
import re
import math as mt
import pandas as pd
import numpy as np
import math
import unicodedata
import sklearn
from numpy import loadtxt
from numpy import save
from numpy import savetxt


def remove_non_ascii(text):
  return unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def frec(text,r):
  fre=[]
  c=0
  for k in range(len(r)):
    fre.append(c)
    c=0
    for i in range(len(text)):
      for j in range(len(text[i,1])):
        if (text[i,1])[j]==(r[k]):
          c=c+1
  return fre

def frecT(text,r):
  fre_tweet=np.zeros([1000, 3581]) 
  c=0
  for i in range(len(text)):
    #c=0
    for k in range(len(r)):
      fre_tweet[i,k]=c
      c=0
      for j in range(len(text[i,1])):
        if (text[i,1])[j]==(r[k]):
          c=c+1
  return fre_tweet

ttext=pd.read_csv("trainTrump.csv")
tweet_text=pd.DataFrame(ttext)
text_data=pd.DataFrame(tweet_text)
data=text_data['Tweets']
data=data.iloc[0:1000]
data_token={}
for i in range(len(data)):
  element=data[i]
  x=[]
  x=element.split(" ")
  data_token[i,1]=x
full_corpus=[]
for i in range(len(data_token)):
  x=data_token[i,1]
  for j in range(len(x)):
    full_corpus.append(x[j])
full_corpus = list(filter(None, full_corpus))
words_bag=list(set(full_corpus))
t=frec(data_token,words_bag)
T=frecT(data_token,words_bag)
N=(len(data_token))
idf=[]
for i in range(len(t)-1):
  ter1=N/t[i+1]
  log=math.log(ter1)
  idf.append(log)
idf.insert(0,0)
my_idf=np.asarray(idf)
wtd=np.multiply(T,my_idf)
wtd=pd.DataFrame(wtd)
data=text_data['label']
data=data.iloc[0:1000]
ar=np.array(data)
savetxt('LabTweets.csv', ar, delimiter=',')
wtd.to_csv('vectorizedTweets.csv')

