'''
Created on 6 oct. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words

en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

lda = models.LdaModel.load('../Save/modelLDA.lda')
dictionary = corpora.Dictionary.load('../Save/scopus_list.dict')
corpus_tfidf = corpora.MmCorpus('../Save/scopus_corpus.mm')
index = similarities.MatrixSimilarity.load('../Save/scopus_research.index')


''' Similarities between pairs of documents '''

similarities_between_pairs = False
if similarities_between_pairs == True:
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
    
    ''' Get topics '''
    
    num_topics = 100
    num_words = 5
    
    pprint(lda.get_topic_terms(40)) # terms for one topic
    #pprint(lda.show_topics(num_topics, num_words))
    #pprint(lda.print_topics())
    #pprint(lda.get_document_topics(corpus_tfidf[68])) # print all topics
    #pprint(lda.get_document_topics(corpus_tfidf[497]))
    #pprint(lda.get_document_topics(corpus_tfidf[1360]))
    
    
    pprint(lda.print_topic(37))
    
    #pprint(lda.print_topic(8))
    #pprint(lda.print_topic(11))
    
    #pprint(dictionary.items())
    #pprint(dictionary.values())

''' Search keys and values '''

''' Per key '''
search_per_key = True
if search_per_key is True:
    searched_key=3454
    p_stemmer = PorterStemmer()
    for key, value in dictionary.items():
        if key == searched_key:
            found_value = value
            print('For key "'+ str(searched_key) +'", the value found is: ' + str(found_value) + '.')
    try:
        found_value
    except NameError:
        print('For key "'+ str(searched_key) +'", no matching value was found.')

''' Per value '''
search_per_value = True
if search_per_value is True:
    searched_value= 'valu'
    stemmed_search_value = p_stemmer.stem(searched_value)
    
    for key, value in dictionary.items():
        if value == stemmed_search_value:
            found_key = key
            print('For value "'+ str(searched_value) +'", the key found is: ' + str(found_key) + '.')
    try:
        found_key
    except NameError:
        print('For value "'+ str(searched_value) +'", no matching key was found.')
        




















