'''
Created on 6 oct. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import numpy as np
import csv
from six import iteritems
import re
from pprint import pprint
import os
import bz2
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from csv import Dialect

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

lda = models.LdaModel.load('../Save/modelLDA.lda')
dictionary = corpora.Dictionary.load('../Save/scopus_list.dict')
corpus_tfidf = corpora.MmCorpus('../Save/scopus_corpus.mm')
index = similarities.MatrixSimilarity.load('../Save/scopus_research.index')


''' Similarities between pairs of documents '''

query = "The challenge of a purposeful design addressed in this article is to align offshore energy systems not only with technical and economic values like efficiency and profitability but also with moral and social values more generally We elaborate a theoretical framework that allows us to make a systematic inventory of embedded values of offshore energy systems and relate them to their societal acceptability By characterizing both objects and subjects of acceptability we shed light on ways to identify areas of value conflicts that must be addressed in purposeful design We suggest the capabilities approach as a normative theory to deal with the arising value conflicts"
vec_bow = dictionary.doc2bow(query.lower().split())
vec_lda = lda[vec_bow]
sims = index[vec_lda]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
#pprint(sims)


''' Get topics '''

#print(lda.get_term_topics(327)) # topics for one term
#pprint(lda.get_topic_terms(0)) # terms for one topic
#pprint(lda.print_topics()) # print all topics
#pprint(lda.get_document_topics(corpus_tfidf[7]))






























