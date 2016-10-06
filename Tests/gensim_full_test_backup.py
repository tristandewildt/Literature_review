'''
Created on 6 oct. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import numpy as np
from stop_words import get_stop_words
import csv
from six import iteritems
import re
from pprint import pprint
import os
import bz2



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)


''' First we import the scopus csv'''
with open('../data/scopus.csv', 'r') as f:
    reader = csv.reader(f)
    scopus_list = list(reader)
    
    
''' Then we clean the data: remove URLs and 'No abstract available',  '''

scopus_list_txt = []

for i in scopus_list:
    i = ''.join(i)
    i = re.sub(r'"' or r'[', ' ', i)
    i = re.sub(r'http\S+', '', i)
    i = re.sub('\W+',' ', i)
    
    i =i.strip("\t").strip("abstract available]")
    scopus_list_txt.append(i)
    
''' Then we save it to a text file '''

f = open('../Save/scopus_list_txt.txt', 'w')
for item in scopus_list_txt:
    f.write(str(item)+'\n')
f.close()

''' Now we create the dictionary'''
dictionary = corpora.Dictionary(line.lower().split() for line in open(r'../Save/scopus_list_txt.txt'))

#stoplist = set('for a of the and to in'.split())
en_stop = get_stop_words('en')
stop_ids = [dictionary.token2id[stopword] for stopword in en_stop
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('../Save/scopus_list.dict')

''' And we create the corpus'''
''' What is done hereunder is a trial. If there are problems in the code or in the quality of the results, problems might come from here.'''

#corpus = [dictionary.doc2bow(line) for line in open(r'../Save/scopus_list_txt.txt')]

class MyCorpus(object):
    def __iter__(self):
        for line in open('../Save/scopus_list_txt.txt'):
            #print(line)
            yield dictionary.doc2bow(line.lower().split())

Scopus_corpus = MyCorpus()
corpora.MmCorpus.serialize('../Save/scopus_corpus.mm', Scopus_corpus)

''' Now documents are transformed from one vector into another. '''

tfidf = models.TfidfModel(Scopus_corpus)
corpus_tfidf = tfidf[Scopus_corpus]

num_topics = 20
num_words = 8

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
corpus_lsi = lsi[corpus_tfidf]
pprint(lsi.show_topics(num_topics, num_words))
lsi.save('../Save/modelLDA.lsi')

#model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics = num_topics)
#pprint(model.show_topics(num_topics, num_words))
#model.save('../Save/modelLDA.lda')

''' Now we look for similarities between pairs of documents '''

doc = "Value"
vec_bow = dictionary.doc2bow(doc.lower().split())
#for i in dictionary:
#    print(i, dictionary[i])
#print(vec_bow)
vec_lsi = lsi[vec_bow]

#index = similarities.MatrixSimilarity(lsi[Scopus_corpus]) # only possible if the total memory required is lower than the RAM. In any other case, you should use similarities.Similarity
#index.save('../Save/scopus_research.index')

#bz2_save = bz2.BZ2Compressor(Scopus_corpus)
#bz2_save.save('../Save/scopus_research.bz2')


































