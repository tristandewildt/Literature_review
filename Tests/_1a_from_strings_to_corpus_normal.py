'''
Created on 23 sep. 2016

@author: tewdewildt
'''
import logging
from gensim import corpora

'''
Link: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

In this example, steps are:
1. Cleaning the document
1.a. Tokenization: segmenting a document into its atomic elements
1.b. Stop words: removing all meaningless words such as 'for' and 'or'
1.c. Stemming: merging words that have equivalent meaning, by reducing them to their common roots. Different methods can be used for that (e.g. the Porter stemming algorithm, the most widely used method), which may differ in aggressivity.
2. Constructing a document-term matrix: to generate an LDA model, we must understand how frequently each term occurs within each document. For that a document-term matrix is created.
Each token (word) is given an unique integer id. Next, the dictionary must be converted into a bag of words. The result is a list of vectors equal to the number of documents. In each document vector is a series of tuples (per token, an id and its frequency in this document).
3. Apply the LDA model.
4. Examine results
'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

''' remove common words and tokenize'''  
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

''' remove words that appear only once'''
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint  # pretty-printer

''' documents are converents to vectors'''
dictionary = corpora.Dictionary(texts)
dictionary.save('../Save/deerwester.dict')  # store the dictionary, for future reference
#print(dictionary.token2id)

''' corpus is saved for later use'''
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('../Save/deerwester.mm', corpus)
pprint(corpus)






