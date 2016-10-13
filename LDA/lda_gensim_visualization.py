'''
Created on 11 oct. 2016

@author: tewdewildt
'''

import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import os
import matplotlib.font_manager
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')


en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

lda = models.LdaModel.load('../Save/modelLDA.lda')
dictionary = corpora.Dictionary.load('../Save/scopus_list.dict')
corpus_tfidf = corpora.MmCorpus('../Save/scopus_corpus.mm')

''' Find word lists '''
n_topics = 10
Find_word_lists = False
if Find_word_lists == True:
    for i in range(1, n_topics):
        temp = lda.show_topic(i, 5)
        terms = []
        for term in temp:
            terms.append(term)
        print("Top 10 terms for topic #" + str(i) + ": "+ ", ".join([i[0] for i in terms]))

''' Make word clouds '''

Make_word_clouds = False
if Make_word_clouds == True:
    for i in range(1, n_topics):
        temp = lda.show_topic(i, 5)
        terms = []
        for term in temp:
            terms.append(term)
    from os import path
    from wordcloud import WordCloud
    
    def terms_to_wordcounts(terms, multiplier=1000):
        return  " ".join([" ".join(int(multiplier*i[1]) * [i[0]]) for i in terms])
    #font_path = os.environ.get("FONT_PATH", "/Library/Fonts/Times New Roman.ttf")
    font_path = "times.ttf"
    wordcloud = WordCloud(font_path, background_color='white').generate(terms_to_wordcounts(terms))
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

''' Topic-words vectors: topics vs. words and PCA'''

Topic_words_vectors_PCA_Topics= True
Topic_words_vectors_PCA_Words= False

from sklearn.feature_extraction import DictVectorizer
    
def topics_to_vectorspace(n_topics, n_words=200):
    rows = []
    for i in xrange(n_topics):
        temp = lda.show_topic(i, n_words)
        row = dict(((i[0],i[1]) for i in temp))
        rows.append(row)
    
    return rows    
    
vec = DictVectorizer()
    
X = vec.fit_transform(topics_to_vectorspace(n_topics))
X.shape
# (40, 2457)

if Topic_words_vectors_PCA_Topics==True:
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    
    X_pca = pca.fit(X.toarray()).transform(X.toarray())
    
    plt.figure()
    for i in xrange(X_pca.shape[0]):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
        plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + str(i))    
    
    plt.title('PCA Topics of keywords: energy and values')
    #plt.savefig("pca_topic")
    
    #plt.show()


if Topic_words_vectors_PCA_Words==True:
    X_pca = pca.fit(X.T.toarray()).transform(X.T.toarray())
    
    plt.figure()
    for i, n in enumerate(vec.get_feature_names()):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
        plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + n, fontsize=8)
        
    plt.title('PCA Words of keywords: energy and values')
    plt.show()

''' hierarchical clustering '''
    
hierarchical_clustering = False
if hierarchical_clustering == True:    
    
    
    plt.figure(figsize=(12,6))
    R = dendrogram(linkage(X_pca))
    plt.show()



''' Correlation matrix '''

correlation_matrix = False
if correlation_matrix == True:
    
    
    cor = squareform(pdist(X.toarray(), metric="euclidean"))
    
    plt.figure(figsize=(12,6))
    R = dendrogram(linkage(cor))
    plt.show()

''' Network '''

network = True
if network == True:
    import networkx as nx

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    
    pca_norm = make_pipeline(PCA(n_components=20), Normalizer(copy=False))
    
    X_pca_norm = pca_norm.fit(X.toarray()).transform(X.toarray())
    
    cor = squareform(pdist(X_pca_norm, metric="euclidean"))
    
    G = nx.Graph()
    
    for i in xrange(cor.shape[0]):
        for j in xrange(cor.shape[1]):
            if i == j:
                G.add_edge(i, j, {"weight":0})
            else:
                G.add_edge(i, j, {"weight":1.0/cor[i,j]})
    
    edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > .8]
    edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G.edges(data=True)])
    
    #pos = nx.graphviz_layout(G, prog="twopi") # twopi, neato, circo
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=.5)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
    #nx.draw_networkx_edge_labels(G, pos ,edge_labels=edge_weight)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.show()






