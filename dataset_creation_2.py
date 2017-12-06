from stop_words import get_stop_words
from gensim import corpora, models, utils
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from pymongo import MongoClient
import logging


def get_tweets(collection):
    return list(collection.find())

def get_seeds(collection):
    query = {
        "isSeed":True
    }
    return list(collection.find(query))


def select_candidates_tweet(collection, dictionary):
    query = {
        "tokens": {
                "$in":dictionary
        },
        "tokens.3":{
            "$exists":True
        },
        "isSeed":False,
        "relevant":{
            "$exists":False
        }
    }
    return list(collection.find(query))

def read_dictionary(dictionaryCollection,previousIteration):
    query = {
        "iteration":previousIteration
    }
    words = list(dictionaryCollection.find(query))
    return list(map(lambda x: x["word"],words))

def evaluate_candidate(doc_model, candidates, tweet_seeds):

    seed_vectors = []
    for t in tweet_seeds:
        vector = doc_model.docvecs[t["label"]]
        seed_vectors.append(vector)

    seed_vectors = np.average(seed_vectors, axis=0)

    similarities = []
    for c in candidates:
        vector = doc_model.docvecs[c["label"]]
        similarity = cosine_similarity(
            vector.reshape(1, -1), seed_vectors.reshape(1, -1))
        
        similarities.append([similarity[0], c["tweet_text"],c["label"]])

    return sorted(similarities, key=lambda x: -x[0])

def annotate_candidates(similarities, lower, higher,collection,iteration_number):
    bulk = collection.initialize_unordered_bulk_op()
    counter = 0
    new_tweet = False

    print(similarities[0])

    for s in similarities:

        similarity = float(s[0][0])

        update_query = {
            "$set": {
                "similarity":similarity,
                "iteration_number": iteration_number
            }
        }
        
        if similarity >= higher:
            update_query["$set"]["relevant"] = True
            new_tweet = True
        elif similarity < lower:
            update_query["$set"]["relevant"] = False
        else:
            continue
        
        print(update_query)
        bulk.find({"label":s[2]}).update(update_query)
        counter+=1

        if (counter % 100 == 0):
            print("Executing a bulk update")
            bulk.execute()
            bulk = collection.initialize_ordered_bulk_op()
    
    if (counter % 100 != 0):
        print("Executing the rest of the bulk")
        bulk.execute()
    
    return new_tweet

def select_candidate_words(collection,model,dictionary,iteration_number):
    query = {
        "relevant":True,
        "iteration_number":iteration_number
    }
    projection = {
        "tokens":1
    }      
    tweets = list(collection.find(query,projection))

    stop_words =  get_stop_words('en')+ get_stop_words('es')

    words = set()

    for t in tweets:
        words = words.union(t["tokens"])

    new_words = []
    
    for w in words:
        print(w)
        if w in dictionary or w in stop_words:
            print(w, "is a dictionary or a stop words")
            continue

        try:
            similar_words = list(map(lambda x: x[0],model.similar_by_word(w,topn=80)))
            if np.any(np.in1d(dictionary,similar_words)):
                new_words.append({
                    "word":str(w),
                    "iteration":iteration_number,
                    "origin":list(set(dictionary).intersection(similar_words))
                })
                print(w,"added to the dictionary")
        except Exception as e:
            print(e)
            continue

    return new_words
    
def setup():
    print("Configuring the logger")
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("Connecting to Mongo")
    client = MongoClient('localhost', 27017)
    db = client['twitter']
    twitterCollection = db["tweet"]

    print("Loading the models")
    word2vec = models.KeyedVectors.load_word2vec_format(
        "GoogleNews-vectors-negative300.bin",binary=True)
    print(word2vec)
    doc2vec = models.Doc2Vec.load("tweet_model_doc2vec.bin")
    print(doc2vec)
   
    twitterCollection = db["tweet"]
    dictionaryCollection = db["dictionary"]
    return dictionaryCollection,twitterCollection, word2vec, doc2vec

def main(iteration_number=50):

    dictionaryCollection, twitterCollection, word2vec, doc2vec = setup()
    for i in range(0,iteration_number):

        print("Starting iteration number ", i)
        dictionary = read_dictionary(dictionaryCollection,i-1)
        

        seeds = get_seeds(twitterCollection)

        print("Retrieving the candidates")
        candidates = select_candidates_tweet(twitterCollection,dictionary)

        print(len(candidates)," candidates found")

        if(len(candidates)==0):
            print("No more candidates found")
            break

        print("Evaluating the candidates")
        similarities = evaluate_candidate(doc2vec,candidates,seeds)

        print("Annotating the candidates")
        are_there_new_tweets = annotate_candidates(similarities,0,0.8,twitterCollection,i)

        new_words = select_candidate_words(twitterCollection,word2vec,dictionary,i)

        if(len(new_words)>0):
            dictionaryCollection.insert_many(new_words)

        if(not are_there_new_tweets and len(new_words)==0):
            print("Cannot further annotate any tweets")
            break

    print("Done in ",i," iterations")

main()
