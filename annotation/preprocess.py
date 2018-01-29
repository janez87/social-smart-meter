# system modules
import re
import sys

# external modules
from stop_words import get_stop_words
from gensim import corpora, models, utils
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# my modules
sys.path.append("../")
import config

TWITTER_COLLECTION = "tweet_ams"
def tokenize(tweet):
    # Remove mentions and url
    tweet_text = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tokens = [token for token in utils.simple_preprocess(
        tweet_text, deacc=False, min_len=3) if token not in stop_words_list]
    return tokens

if __name__=="__main__":
    stop_words_list = get_stop_words('en') + get_stop_words('nl')

    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]
    twitterCollection = db[TWITTER_COLLECTION]

    bulk = twitterCollection.initialize_ordered_bulk_op()
    counter = 0

    print("Processing the tweets")

    tweets = list(twitterCollection.find())
    for tweet in tweets:
       
        if(tweet["truncated"]):
            text = tweet["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]

        tokens = tokenize(text)
        bulk.find({'_id': tweet['_id']}).update({'$set': {'tokens': tokens,"label":tweet["id_str"]}})
        counter += 1

        #if counter % 1000 == 0:
        #    print("Saving a bulk ",counter)
        #    bulk.execute()
        #    bulk = db.coll.initialize_ordered_bulk_op()
            

    if counter % 1000 != 0:
        print("Saving the rest")
        bulk.execute()

    print("Done")
