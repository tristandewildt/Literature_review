'''
Created on 23 sep. 2016

@author: tewdewildt
'''
''' This code is a memory friendly version of from strings to corpus '''

import logging
import gensim
from gensim import corpora
from pprint import pprint
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


''' optionally a new dictionary is constructed without loading all texts into memory '''
    
from six import iteritems
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('../data/mycorpus.txt'))
# remove stop words and words that appear only once
stoplist = set('for a of the and to in'.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('../Save/deerwester.dict')
#print(dictionary)

''' optionally a new corpus is created that uses less memory (doesn't load the corpus into memory) '''
class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
corpora.MmCorpus.serialize('../Save/corpus_memory_friendly.mm', corpus_memory_friendly)
#print(corpus_memory_friendly)

all_vectors = [vector for vector in corpus_memory_friendly]
#pprint(all_vectors)

#numpy_matrix = np.random.randint(10, size=[5,2])  # random matrix as an example
#corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
#numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=5)
#print(numpy_matrix)



