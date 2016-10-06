'''
Created on 23 sep. 2016

@author: tewdewildt
'''

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import csv
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

with open('../data/scopus.csv', 'r') as f:
    reader = csv.reader(f)
    scopus_list = list(reader)

def remove_urls(vTEXT):
    vTEXT = re.sub(r'^https?:\/\/.*[\r\n]*', '', vTEXT, flags=re.MULTILINE)

for i in scopus_list:
    
    i = ''.join(i)
    i = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', i)
    ''' Still a problem here with removing the first word of the abstract '''
    
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

   
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

num_topics = 10
num_words = 4
## generate LDA model
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=2)

'''To get list of topics'''
#print(ldamodel.print_topics(num_topics, num_words))

#ldamodel2 = gensim.corpora.dictionary
#print(ldamodel.print_topic(0, topn=10))

'''To get list of topics per document'''
#print(ldamodel.get_document_topics(1, minimum_probability=None, minimum_phi_value=None, per_word_topics=False))

'''To get most likely topics for a particular word in vocab'''
#print(ldamodel.get_term_topics(0, minimum_probability=None))

'''To get most likely topics for a particular word in vocab'''
#print(ldamodel.get_topic_terms(0, topn=10))


















