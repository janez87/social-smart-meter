# system modules
import sys
import logging
import math

# external modules
from gensim import models
from pymongo import MongoClient
from sklearn import svm

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
    doc2vec = models.Doc2Vec.load("../models/tweet_model_doc2vec_v2.bin")
    print(doc2vec)

    twitterCollection = db["tweet"]
    dictionaryCollection = db["dictionary"]

    return dictionaryCollection, twitterCollection, doc2vec


def get_training_test_sets(collection,training_ratio,doc_model):
    n_total = collection.find({"relevant":{"$exists":True}}).count()

    n_training = math.floor(n_total*training_ratio) 
    
    training_set = list(collection.aggregate([{"$match":{"relevant": {"$exists": True}}},{"$sample":{"size":n_training}}]))

    training_set_ids = list(map(lambda x: x["_id"],training_set))

    test_set = list(collection.find(
        {"_id": {"$nin": training_set_ids}, "relevant": {"$exists": True}}))
    
    print(len(training_set))
    print(len(test_set))

    training_set = list(map(lambda x: {
        "label":x["label"],
        "vector": doc_model.docvecs[x["label"]],
        "relevant":x["relevant"]
    },training_set))

    test_set = list(map(lambda x: {
         "label": x["label"],
         "vector": doc_model.docvecs[x["label"]],
         "gt":x["crowd_evaluation"]
    },test_set))

 
    return training_set, test_set

def train(training_set):
    clf = svm.SVC(kernel='linear', C=1.0)

    X=[]
    Y=[]

    for t in training_set:
        X.append(t["vector"])
        Y.append(t["relevant"])

    clf.fit(X,Y)

    return clf

def test(model, test_set):

    TP = 0
    FP = 0
    FN = 0
    for t in test_set:
        t["predicted"] = model.predict(t["vector"].reshape(1, -1))
        TP += t["predicted"]*t["gt"]
        FP += t["predicted"]*(1-t["gt"])
        FN += t["gt"]*(1-t["predicted"])

    precision = TP/(TP+FP)
    recall = TP / (TP + FN)
    print("Precision: ",precision)
    print("Recall: ",recall)

def main():
    dictionaryCollection, twitterCollection, doc2vec = setup()


    training, test_set = get_training_test_sets(twitterCollection,0.8,doc2vec)

    clf = train(training)

    test(clf,test_set)

main()
