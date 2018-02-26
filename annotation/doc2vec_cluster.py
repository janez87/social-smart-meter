# system modules
import sys
# external modules
from gensim import corpora, models
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pymongo import MongoClient
import matplotlib.pyplot as plt

# my modules
sys.path.append("../")
import config


def get_tagged_documents(collection, doc_model):
    query = {
        "tokens.3": {
            "$exists": True
        }
    }
    tweets = list(collection.find(query))
    tweets = list(map(lambda x: {
        "label": x["label"],
        "vector": doc_model.docvecs[x["label"]],
        "tokens": x["tokens"]
    }, tweets))
    return tweets


def cluster(X, k):
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    dist = sum(np.min(
        cdist(X, kmean.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]

    return dist, kmean


if __name__ == "__main__":
    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]
    twitterCollection = db["tweet"]

    print("Loading the model")
    doc2vec = models.Doc2Vec.load("../models/tweet_model_doc2vec_v2.bin")
    distortions = []
    tweets = get_tagged_documents(twitterCollection, doc2vec)
    X = list(map(lambda x: x["vector"], tweets))
    X = np.array(X)

    dist, kmean = cluster(X, 20)

    centroids = kmean.cluster_centers_

    centroids_documents = [doc2vec.docvecs.most_similar(
        [c], topn=1) for c in centroids]

    print(centroids_documents)
