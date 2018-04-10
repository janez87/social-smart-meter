# system modules
import sys

# external modules
from gensim import models
from pymongo import MongoClient
from random import shuffle

# my modules
sys.path.append("../")
import config


def get_tagged_documents(collection):
    query = {
        "tokens.3": {
            "$exists": True
        }
    }

    tweets = list(collection.find(query))

    return list(
        map(lambda x: models.doc2vec.TaggedDocument(x["tokens"], [str(x["label"])]), tweets))


def get_words(collection):
    query = {
        "tokens.3": {
            "$exists": True
        }
    }
    tweets = list(collection.find(query))

    return list(map(lambda x: x["tokens"], tweets))


def train_doc_model(corpus, file):
    print("Training Doc2Vec model")
    model = models.Doc2Vec(corpus, size=100)
    model.save(file)


def train_doc_model_manual(corpus, file):
    print("Training Dord2Vec model")
    model = models.Doc2Vec(dm=1, iter=5, alpha=0.1,min_alpha=0.025, size=100)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(file)


def train_word_model(corpus, file):
    print("Training Word2Vec model")
    model = models.Word2Vec(corpus)
    model.save(file)


print("Connecting to Mongo")
client = MongoClient(config.DB_HOST, config.DB_PORT)
db = client[config.DB_NAME]

TWEET_COLLECTION = "tweet"
twitterCollection = db[TWEET_COLLECTION]

tweets = get_tagged_documents(twitterCollection)
shuffle(tweets)
train_doc_model(tweets, "../models/tweet_model_doc2vec_v2_100_new.bin")

# words = get_words(twitterCollection)
# train_word_model(words, "tweet_model_word2vec_amsterdam.bin")
