'''
Created on 23 sep. 2016

@author: tewdewildt
'''
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

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
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
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

num_topics = 2
num_words = 4
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=200)

#print(ldamodel.print_topics(num_topics, num_words))
print(type(doc_set))