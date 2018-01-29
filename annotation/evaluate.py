# system modules
import logging
import sys

# external modules
import numpy as np
from pymongo import MongoClient

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
    twitterCollection = db["tweet"]

    dictionaryCollection = db["dictionary_2"]


    return dictionaryCollection, twitterCollection

def get_TP_rate(tweetCollection):
    total_positive = tweetCollection.find({"crowd_evaluation": True}).count()
    TP = tweetCollection.find(
        {"relevant":True,"crowd_evaluation":True}).count()

    return TP / total_positive

def get_FP_rate(tweetCollection):
    total = tweetCollection.find({"crowd_evaluation": False}).count()
    FP = tweetCollection.find(
        {"relevant": True, "crowd_evaluation": False}).count()

    return FP / total

def get_TN_rate(tweetCollection):
    total = tweetCollection.find({"crowd_evaluation": False}).count()
    TN = tweetCollection.find(
        {"relevant": False, "crowd_evaluation": False}).count()

    return TN / total

def get_FN_rate(tweetCollection):
    total = tweetCollection.find({"crowd_evaluation": True}).count()
    FN = tweetCollection.find(
        {"relevant": False, "crowd_evaluation": True}).count()

    return FN / total

def get_precision(tweetCollection):
    TP = tweetCollection.find(
       {"relevant": True, "crowd_evaluation": True}).count()
    FP = tweetCollection.find(
         {"relevant": True, "crowd_evaluation": False}).count()
   
   #TP / TP+FP
    return TP/(TP+FP)

def get_recall(tweetCollection):
    TP = tweetCollection.find(
        {"relevant": True, "crowd_evaluation": True}).count()
    FN = tweetCollection.find(
        {"relevant": False, "crowd_evaluation": True}).count()

    return TP/(TP+FN)

''' I consider this as the baseline because if I suppose to not have the seeds, 
I have to consider as valid a tweet if it contains a dictionary word, so it has been considered by the pipeline'''
def get_baseline_precision(tweetCollection):
    TP = tweetCollection.find({"relevant":{"$exists":True},"crowd_evaluation":True}).count()
    FP = tweetCollection.find({"relevant":{"$exists":True},"crowd_evaluation":False}).count()

    return TP / (TP + FP)

def main():
    dictionary, twitter = setup()

    TP_rate = get_TP_rate(twitter)  
    FP_rate = get_FP_rate(twitter)
    TN_rate = get_TN_rate(twitter)
    FN_rate = get_FN_rate(twitter)

    precision = get_precision(twitter)
    recall = get_recall(twitter)

    print("TP rate ",TP_rate)
    print("FP rate ",FP_rate)
    print("TN rate ",TN_rate)
    print("FN rate ",FN_rate)
    print("precision",precision)
    print("recall",recall)

    print("precision baseline", get_baseline_precision(twitter))
main()
