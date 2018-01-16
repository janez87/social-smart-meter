from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics 
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
en_stop.append("rt")

p_stemmer = PorterStemmer()

with open('out3.txt', 'r') as f:
    tweet_list = list(f.readlines())

texts = []
for tweet in tweet_list:
    tweet = tweet.lower()

    tokens = tokenizer.tokenize(tweet)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    if(len(stopped_tokens)==0):
        continue
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stopped_tokens)

#model = models.Word2Vec(texts,min_count=1)
model = models.KeyedVectors.load_word2vec_format(
    '/Users/andreamauri/Documents/Develop/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',binary=True)

X = []

for tweet in texts:
    tweet_vector = []
    for w in tweet:
        try:
            tweet_vector.append(model.wv[w])
        except KeyError as e:
            #print(e)
            continue 
    tweet_vector = np.sum(tweet_vector,axis=0)  
    
    if not isinstance(tweet_vector, (list, tuple, np.ndarray)):
        continue
    
    tweet_vector = normalize(tweet_vector.reshape(1,-1))
    #print(tweet_vector)
    X.append(tweet_vector)

X = np.array(X)

print(X)

from sklearn.metrics.pairwise import cosine_similarity

K = range(2, 41)
silhouette = []
for k in K:
    print("Clustering with k = " + str(k))
    sc = SpectralClustering(k, assign_labels='discretize',affinity=cosine_similarity)
    sc.fit(X)
    silhouette.append(metrics.silhouette_score(
        X, sc.labels_, metric='euclidean'))

print(silhouette)
"""
distortions = []
silhouette = []
K = range(2, 21)
for k in K:
    print("Clustering with k = " + str(k))
    kmeanModel = KMeans(n_clusters=k, init='k-means++').fit(X)
    kmeanModel.fit(X)
    print("Computing the errors")
    distortions.append(sum(np.min(
        cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print("Computing the average silouette score ")
    silhouette.append(silhouette_score(
        X, kmeanModel.labels_, metric='euclidean'))

print(silhouette)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

"""

