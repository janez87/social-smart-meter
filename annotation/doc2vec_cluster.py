# system modules
import sys
# external modules
from gensim import corpora, models
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from pymongo import MongoClient
import matplotlib.pyplot as plt
#my modules
sys.path.append("../")
import config


def get_tagged_documents(collection,doc_model):
    query = {
        "tokens.3": {
            "$exists": True
        }
    }
    tweets = list(collection.find(query))
    tweets = list(map(lambda x: {
         "label": x["label"],
         "vector": doc_model.docvecs[x["label"]],
         "tokens":x["tokens"]
    }, tweets))
    return tweets

def cluster(X,k):
    kmean = DBSCAN(eps=k)
    kmean.fit(X)
    dist = sum(np.min(
        cdist(X, kmean.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]

    return dist


if __name__ == "__main__":
    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]
    twitterCollection = db["tweet"]

    print("Loading the model")
    doc2vec = models.Doc2Vec.load("../models/tweet_model_doc2vec_v2.bin")
    distortions = []
    tweets = get_tagged_documents(twitterCollection,doc2vec)
    X = list(map(lambda x: x["vector"],tweets))
    X = np.array(X)
    for i in np.arange(0.3, 0.5, 0.1):
        print("K = ",i)
        distortions.append(cluster(X,i)) 
    
    plt.plot(np.arange(0.3, 0.5, 0.1), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

