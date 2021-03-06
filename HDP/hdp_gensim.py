'''
Created on 18 okt. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models
import csv
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
from six import iteritems
import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


''' First we import the scopus csv'''
with open('../data/scopus_value_smart_grid.csv', 'r') as f:
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
#dictionary.save_as_text('../Save/scopus_list_dict.txt')

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

Scopus_corpus = corpora.MmCorpus('../Save/scopus_corpus.mm')
tfidf = models.TfidfModel(Scopus_corpus)
corpus_tfidf = tfidf[Scopus_corpus]

hdp = gensim.models.hdpmodel.HdpModel(Scopus_corpus, id2word = dictionary)
hdp.optimal_ordering()
hdp.save('../Save/modelHDP.hdp')
x = hdp.show_topics(1000, 10)
print(x)

''' Topics can also be formatized (does not work yet) '''
#hdp_formatter = gensim.models.hdpmodel.HdpTopicFormatter(dictionary=dictionary, topic_data=Scopus_corpus)
#hdp_formatter.print_topics(10, 10)

















