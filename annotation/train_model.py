# system modules
import sys

# external modules
from gensim import models
from pymongo import MongoClient

# my modules
sys.path.append("../")
import config



def get_tagged_documents(collection):
    query = {
        "tokens.3":{
            "$exists":True
        }
    }
    tweets = list(collection.find(query))

    return list(map(lambda x: models.doc2vec.TaggedDocument(x["tokens"],[x["label"]]),tweets))

def get_words(collection):
    query = {
        "tokens.3": {
            "$exists": True
        }
    }
    tweets = list(collection.find(query))

    return list(map(lambda x: x["tokens"], tweets))

def train_doc_model(corpus,file):
    print(corpus[0])
    print("Training Doc2Vec model")
    model = models.Doc2Vec(corpus)
    model.save(file)


def train_word_model(corpus,file):
    print("Training Word2Vec model")
    model = models.Word2Vec(corpus)
    model.save(file)

print("Connecting to Mongo")
client = MongoClient(config.DB_HOST, config.DB_PORT)
db = client[config.DB_NAME]
twitterCollection = db["tweet"]

tweets = get_tagged_documents(twitterCollection)
train_doc_model(tweets, "tweet_model_doc2vec_v2.bin")

words = get_words(twitterCollection)
train_word_model(words,"tweet_model_word2vec.bin")
