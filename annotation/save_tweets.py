# system modules
import re
import sys

# external modules
from stop_words import get_stop_words
from gensim import corpora, models, utils
from pymongo import MongoClient

# my modules
sys.path.append("../")
import config

tweet_seeds_file = "../tweet_usa_set.txt"
tweet_list = []

stop_words_list = get_stop_words('en') + get_stop_words('es')

with open(tweet_seeds_file, "r") as tweets:
        for i, tweet in enumerate(tweets):
            tweet_text = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
            if(tweet_text.startswith('"')):
                #fast hack for removing probelmatic tweets (~28k on ~200k)
                continue

            label = i
            tokens = [token for token in utils.simple_preprocess(
                tweet_text, deacc=False, min_len=3) if token not in stop_words_list]
            
            print(tokens)
            tweet_list.append({
                "tweet_text":tweet_text,
                "tokens":tokens,
                "isSeed":False,
                "label":label
            })

print("Connecting to Mongo")
client = MongoClient(config.DB_HOST, config.DB_PORT)
db = client[config.DB_NAME]
twitterCollection = db["tweet"]

result = twitterCollection.insert_many(tweet_list)

print(result)
