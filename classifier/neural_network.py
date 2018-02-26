# system modules
import sys
import logging
import math

# external modules
from gensim import models
from pymongo import MongoClient
import numpy as np
import tensorflow as tf
import pandas as pd
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

    return dictionaryCollection, twitterCollection, doc2vec


def get_training_test_sets(collection, training_ratio, doc_model):
    n_total = collection.find({"relevant": {"$exists": True}}).count()

    n_training = math.floor(n_total * training_ratio)

    training_set = list(collection.aggregate(
        [{"$match": {"relevant": {"$exists": True}}}, {"$sample": {"size": n_training}}]))

    training_set_ids = list(map(lambda x: x["_id"], training_set))

    test_set = list(collection.find(
        {"_id": {"$nin": training_set_ids}, "crowd_evaluation": {"$exists": True}}))

    print(len(training_set))
    print(len(test_set))

    training_set = list(map(lambda x: {
        "label": x["label"],
        "vector": doc_model.docvecs[str(x["label"])],
        "relevant": x["relevant"]
    }, training_set))

    test_set = list(map(lambda x: {
        "label": x["label"],
        "vector": doc_model.docvecs[str(x["label"])],
        "gt": x["crowd_evaluation"]
    }, test_set))

    return training_set, test_set

def feed_training_set(inputs,outputs,size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(inputs),outputs))
    dataset = dataset.shuffle(1000).repeat().batch(size)
    return dataset.make_one_shot_iterator().get_next()


def evaluate(attributes, classes, batch_size):
    attributes = dict(attributes)
    if classes is None:
        inputs = attributes
    else:
        inputs = (attributes, classes)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
     
if __name__ == "__main__":

    dictionaryCollection, twitterCollection, doc2vec = setup()

    training, test = get_training_test_sets(
        twitterCollection, 1, doc2vec)

    training_x = pd.DataFrame(np.array([x["vector"] for x in training]),columns=[str(x) for x in range(0,300)])
    training_y = np.array([x["relevant"] for x in training])  

    print(training_x)
    assert training_x.shape[0] == training_y.shape[0]

    test_x = pd.DataFrame(np.array([x["vector"] for x in test]), columns=[
                          str(x) for x in range(0, 300)])
    test_y = np.array([x["gt"] for x in test])

    assert test_x.shape[0] == test_y.shape[0]

    c = [tf.feature_column.numeric_column(key=str(x)) for x in range(0,300) ]
   
    classifier = tf.estimator.DNNClassifier(
        feature_columns = c,
        # Two hidden layers of 10 nodes each.
        hidden_units=[200, 200],
        # The model is classifying 3 classes
        n_classes=2)

    classifier.train(
        input_fn=lambda: feed_training_set(training_x, training_y, 100),
        steps=1000)

    eval_result = classifier.evaluate(
        input_fn=lambda: evaluate(test_x, test_y, 100))
