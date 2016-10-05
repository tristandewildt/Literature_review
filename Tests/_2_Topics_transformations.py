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
Goal of this tutorial is to show how to transform documents form one vector representations to another. This serves two goals:
1. Bring out the hidden structure in the corpus, discover relationships between words and use them to described the documents in a new and (hopefully) more semantic way.
2. Make the document representation more compact. 
'''


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

''' First the saved dictionary is used'''
if (os.path.exists('../Save/deerwester.dict')):
    dictionary = corpora.Dictionary.load('../Save/deerwester.dict')
    corpus = corpora.MmCorpus('../Save/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

''' First a transformation is created, which are standard Python objects. Different transformations may however require different initialization parameters.'''

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]

#pprint(lsi.print_topics(2))

all_docs = [doc for doc in corpus_lsi]
#pprint(all_docs)

lsi.save('../Save/model.lsi')

''' Different types of transformations are available
https://radimrehurek.com/gensim/tut2.html
Next to the ones proposed here, may also look at 'Term Frequency * Inverse Document Frequency', 'Random Projections' 
'''

num_topics = 2
num_words = 4

''' 1. Latent Semantic Indexing (LSI) 
Used to analyse relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms'''

model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
pprint(model.show_topics(num_topics, num_words))

''' 2. Latent Dirichlet Allocation (LDA) 
Also a transformation from bag-of-words counts into a topic space of lower dimensionality. Uses a probabilistic extension (of LSA) so LDA's topics can be interpreted as probability distributions over words'''

model = models.LdaModel(corpus, id2word=dictionary, num_topics=2)
pprint(model.show_topics(num_topics, num_words))
model.save('../Save/modelLDA.lda')

''' 3. Hierarchical Dirichlet Process (HDP)
A nonparametric Bayesian approach to clustering grouped data '''

model = models.HdpModel(corpus, id2word=dictionary)
pprint(model.show_topics(num_topics, num_words))









