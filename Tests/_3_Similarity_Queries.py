'''
Created on 23 sep. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import os
import bz2

''' 
The reason of creating a corpus is that we want to use it to determine similarity between pairs of documents, 
or the similarity between a specific document and a set of other documents (such as a user query vs. indexed documents).
'''

dictionary = corpora.Dictionary.load('../Save/deerwester.dict')
corpus = corpora.MmCorpus('../Save/deerwester.mm')
#print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

''' The following code checks whether words of a certain query are present in the dictionary previously created '''

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
#for i in dictionary:
#    print(i, dictionary[i])
#print(vec_bow)
vec_lsi = lsi[vec_bow]
#print(vec_lsi)

''' Then the query structures are initialized. The index is based on the corpus created earlier and is used for comparison '''

index = similarities.MatrixSimilarity(lsi[corpus]) # only possible if the total memory required is lower than the RAM. In any other case, you should use similarities.Similarity
index.save('../Save/deerwester.index')

bz2_save = bz2.BZ2Compressor(corpus)
bz2_save.save('../Save/deerwester.bz2')
#index = similarities.MatrixSimilarity.load('/tmp/deerwester.index') # to load the index

''' Now queries are performed: this allows to obtain similarities of our query document against the indexed docuements''' 
# idea might be here, for later, to take one document of which we are sure that it lies in the same topic of interests, and check which others are similar to it. Could also do this based on literature cited
#(not by co-citation directly but by taking words in the bibliography direction, which may point to similarities in authors and journal names. This might be quicker than google cleaner and possible as effective.
sims = index[vec_lsi]
print(list(enumerate(sims))) 
# the range is between -1 and 1; the greater, the most similar.

''''This list might also be sorted'''
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

''' The thing to note here is that documents no. 2 ("The EPS user interface management system") and 4 ("Relation of user perceived response time to error measurement") 
would never be returned by a standard boolean fulltext search, because they do not share any common words with "Human computer interaction".
However, after applying LSI, we can observe that both of them received quite high similarity scores (no. 2 is actually the most similar!), 
which corresponds better to our intuition of them sharing a 'computer-human' related topic with the query. 
In fact, this semantic generalization is the reason why we apply transformations and do topic modelling in the first place. '''


