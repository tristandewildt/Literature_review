'''
Created on 23 sep. 2016

@author: tewdewildt
'''

import logging, gensim

logging.basicConfig(
                    format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_output_wordids.txt')