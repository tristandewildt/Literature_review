'''
Created on 23 sep. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import os

''' 
The reason of creating a corpus is that we want to use it to determine similarity between pairs of documents, 
or the similarity between a specific document and a set of other documents (such as a user query vs. indexed documents).
'''

dictionary = corpora.Dictionary.load('../Save/deerwester.dict')
corpus = corpora.MmCorpus('../Save/deerwester.mm')
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

''' The following code checks whether words of a certain query are present in the dictionary previously created '''

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
#for i in dictionary:
#    print(i, dictionary[i])
print(vec_bow)
vec_lsi = lsi[vec_bow]
print(vec_lsi)

''' Then the query structures are initialized '''






