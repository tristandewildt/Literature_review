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
import os
import bz2
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from csv import Dialect
from itertools import islice



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


''' First we import the scopus csv'''
with open('../data/scopus11.csv', 'r') as f:
    #next(f)
    reader = csv.reader(f)   
    scopus_list = list(reader)

''' Then we clean the data: remove URLs and 'No abstract available',  '''

scopus_list_txt = []

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

t = open('../Save/scopus_list_no_tolken.txt', 'w')

for i in scopus_list:
    i = ''.join(i)
    i = re.sub(r'"' or r'[', ' ', i)
    i = re.sub(r'http\S+', '', i)
    i = re.sub('\W+',' ', i)
    i =i.strip("\t").strip("abstract available]")
    
    t.write(str(i)+'\n')
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

#Then we remove stop words and words that only appear once

    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    scopus_list_txt.append(stemmed_tokens)
    


''' Then we save it to a text file '''
f = open('../Save/scopus_list_txt.txt', 'w')
for item in scopus_list_txt:
    f.write(str(item)+'\n')
f.close()

#print(scopus_list_txt[68])


''' Now we create the dictionary'''
dictionary = corpora.Dictionary(scopus_list_txt)
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once
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

corpus = [dictionary.doc2bow(text) for text in scopus_list_txt]

corpora.MmCorpus.serialize('../Save/scopus_corpus.mm', corpus)

''' Now documents are transformed from one vector into another. '''

Scopus_corpus = corpora.MmCorpus('../Save/scopus_corpus.mm')
tfidf = models.TfidfModel(Scopus_corpus)
corpus_tfidf = tfidf[Scopus_corpus]


num_topics = 10
num_words = 5


# topics = 20 and passes = 1000 seems to provide interesting results.

#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
#corpus_lsi = lsi[corpus_tfidf]
#print(lsi.show_topics(num_topics, num_words))
#lsi.save('../Save/modelLDA.lsi')

lda = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics, id2word = dictionary, passes=200)
#lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics = num_topics)
#pprint(lda.show_topics(num_topics, num_words))
lda.save('../Save/modelLDA.lda')

''' Now we look for similarities between pairs of documents '''

query = "The challenge of a purposeful design addressed in this article is to align offshore energy systems not only with technical and economic values like efficiency and profitability but also with moral and social values more generally We elaborate a theoretical framework that allows us to make a systematic inventory of embedded values of offshore energy systems and relate them to their societal acceptability By characterizing both objects and subjects of acceptability we shed light on ways to identify areas of value conflicts that must be addressed in purposeful design We suggest the capabilities approach as a normative theory to deal with the arising value conflicts"
split_lower_query = query.lower().split()
stopped_query = [f for f in split_lower_query if not f in en_stop]
stemmed_query = [p_stemmer.stem(h) for h in stopped_query]

vec_bow = dictionary.doc2bow(stemmed_query)
vec_lda = lda[vec_bow]

index = similarities.MatrixSimilarity(lda[corpus_tfidf]) # only possible if the total memory required is lower than the RAM. In any other case, you should use similarities.Similarity
index.save('../Save/scopus_research.index')

sims = index[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
pprint(sims)



#bz2_save = bz2.BZ2Compressor(Scopus_corpus)
#bz2_save.save('../Save/scopus_research.bz2')


































