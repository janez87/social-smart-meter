# system modules
import sys
import logging
import math
import random

# external modules
from gensim import models
from pymongo import MongoClient
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import vstack, csr_matrix
import numpy as np

# my modules
sys.path.append("../")
import config


def setup():
    print("Configuring the logger")
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]

    print("Loading the models")
    doc2vec = models.Doc2Vec.load("../models/tweet_model_doc2vec_v2_300.bin")

    twitterCollection = db["tweet"]
    dictionaryCollection = db["dictionary"]

    vectorizer = HashingVectorizer(stop_words='english', ngram_range=(1, 1))

    documents = list(twitterCollection.find())
    documents = list(map(lambda x: (' '.join(x["tokens"])),documents))
    vectorizer.fit(documents)

    return dictionaryCollection, twitterCollection, doc2vec, vectorizer


def get_training_test_sets(collection, training_ratio, doc_model, vectorizer):
    n_total = collection.find({"relevant": {"$exists": True}}).count()

    n_training = math.floor(n_total * training_ratio/2)

    training_set = list(collection.aggregate([{"$match": {"relevant": True}}, {"$sample": {"size": n_training}}]))
    training_set += list(collection.aggregate([{"$match": {"relevant": False}}, {"$sample": {"size": n_training}}]))

    training_set_ids = list(map(lambda x: x["_id"], training_set))

    test_set = list(collection.find(
        {"_id": {"$nin": training_set_ids}, "crowd_evaluation": {"$exists": True}}))

    #print(len(training_set))
    #print(len(test_set))

    training_set = list(map(lambda x: {
        "label": x["label"],
        #"vector": vectorizer.transform([' '.join(x["tokens"])]),
        "vector": doc_model.docvecs[str(x["label"])],
        "relevant": x["relevant"]
    }, training_set))

    test_set = list(map(lambda x: {
        "label": x["label"],
        #"vector": vectorizer.transform([' '.join(x["tokens"])]),
        "vector":doc_model.docvecs[str(x["label"])],
        "gt": x["crowd_evaluation"],
        "text": x["text"]
    }, test_set))

    return training_set, test_set


def train(training_set,C,gamma):
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    #X = csr_matrix((0,1048576))
    X = []
    Y = []

    for t in training_set:
        normalized_vector = normalize(t["vector"].reshape(1, -1), axis=1)[0]
        X.append(normalized_vector)
        #X = vstack([X,t["vector"]])
        Y.append(t["relevant"])

    clf.fit(X, Y)

    return clf


def test(model, test_set):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for t in test_set:
        normalized_vector = normalize(t["vector"].reshape(1, -1), axis=1)
        t["predicted"] = model.predict(normalized_vector)
        # t["confidence"]=model.decision_function(
        #   normalized_vector.reshape(1, -1))[0]
        TP += t["predicted"] * t["gt"]
        FP += t["predicted"] * (1 - t["gt"])
        FN += t["gt"] * (1 - t["predicted"])
        TN += (1 - t["predicted"]) * (1 - t["gt"])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    '''print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("Precision", precision)
    print("Recall ", recall)
    print("Accuracy ", accuracy)'''

    return precision, recall, accuracy


def main():
    dictionary_collection, twitter_collection, doc2vec, vectorizer = setup()

    C = [1,10,100,1000,10000]
    #gammas = [1,0.1,0.01,0.001,0.0001]
    gammas = [0.01]
    print("Evaluating svm")

    for c in C:

        for g in gammas:
            precisions = []
            recalls = []
            accuracies = []

            print('C: ', c, 'Gamma: ', g)

            for i in range(0, 50):
                #print("Run ", i)
                training, test_set = get_training_test_sets(twitter_collection, 0.8, doc2vec, vectorizer)

                clf = train(training,c,g)

                p, r, a = test(clf, test_set)
                precisions.append(p)
                recalls.append(r)
                accuracies.append(a)

            #print('C: ', c[k], 'Gamma: ', gammas[k])
            print("Precision: ", np.average(precisions), " std: ", np.std(precisions))
            print("Recall: ", np.average(recalls), " std: ", np.std(recalls))
            print("Accuracy: ", np.average(accuracies), " std: ", np.std(accuracies))


main()
