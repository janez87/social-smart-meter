# system modules
import logging
import sys

# external modules
from gensim import models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from pymongo import MongoClient
from translation import bing

# my modules
sys.path.append("../")
import config


def get_tweets(collection):
    return list(collection.find())


def get_seeds(collection):
    ''' query = {
        "$or":[{"isSeed":True},{"relevant":True}]
    } '''
    query = {
        "isSeed": True
    }
    return list(collection.find(query))


def to_english(tokens, src_lang):
    sentence = " ".join(tokens)
    english_sentence = bing(sentence, src=src_lang, dst="en")
    print(english_sentence)
    return english_sentence.split()


def get_tweet_vector(tweet, word_model):
    word_vectors = []

    ''' if(tweet["lang"] != "en" and tweet["lang"] != "und"):
        tokens = to_english(tweet["tokens"],tweet["lang"])
    else:
        tokens = tweet["tokens"] '''

    tokens = tweet["tokens"]
    for t in tokens:
        if t in word_model:
            word_vectors.append(word_model[t])
        else:
            print(t)

    if (len(word_vectors) == 0):
        return []

    vector = np.average(word_vectors, axis=0)

    return vector


def select_candidates_tweet(collection, dictionary):
    query = {
        "tokens": {
            "$in": dictionary
        },
        "tokens.3": {
            "$exists": True
        },
        "relevant": {
            "$exists": False
        },
        "$or": [{
            "isSeed": {"$exists": False}
        }, {"isSeed": False}]
    }
    return list(collection.find(query))


def read_dictionary(dictionary_collection, previous_iteration):
    query = {
        "iteration": previous_iteration
    }
    words = list(dictionary_collection.find(query))
    return list(map(lambda x: x["word"], words))


def evaluate_candidate(doc_model, candidates, tweet_seeds, word_model):
    seed_vectors = []
    for t in tweet_seeds:
        vector = doc_model.docvecs[t["label"]]
        # vector = normalize(vector.reshape(1, -1), axis=1)[0]
        # vector = doc_model.infer_vector(t["tokens"])
        # vector = get_tweet_vector(t,word_model)
        if len(vector) > 0:
            seed_vectors.append(vector)

    seed_vectors = np.average(seed_vectors, axis=0)
    # seed_vectors = normalize(seed_vectors.reshape(1, -1), axis=1)[0]
    # seed_vectors = np.array(seed_vectors)
    similarities = []
    for c in candidates:
        vector = doc_model.docvecs[c["label"]]
        # vector = normalize(vector.reshape(1, -1), axis=1)[0]
        # vector = doc_model.infer_vector(c["tokens"])
        # vector = get_tweet_vector(c, word_model)
        ''' if(len(vector) > 0):
            similarity = cosine_similarity(
                vector.reshape(1, -1), seed_vectors.reshape(1, -1))
        else:
            similarity = [[-1.0]] '''

        similarity = cosine_similarity(
            vector.reshape(1, -1), seed_vectors.reshape(1, -1))
        similarities.append([similarity[0], c["tokens"], c["label"]])

    return similarities


def annotate_candidates(similarities, lower, higher, collection, iteration_number):
    bulk = collection.initialize_ordered_bulk_op()
    counter = 0
    new_tweets = 0

    for s in similarities:

        print(s)
        similarity = float(s[0])

        update_query = {
            "$set": {
                "similarity": similarity,
                "iteration_number": iteration_number
            }
        }

        if similarity >= higher:
            update_query["$set"]["relevant"] = True
            new_tweets += 1
        elif similarity < lower:
            update_query["$set"]["relevant"] = False
        else:
            continue

        print(update_query)
        bulk.find({"label": s[2]}).update(update_query)
        counter += 1

        if counter % 100 == 0:
            print("Executing a bulk update")
            bulk.execute()
            bulk = collection.initialize_ordered_bulk_op()

    if counter % 100 != 0:
        print("Executing the rest of the bulk")
        bulk.execute()

    return new_tweets


def select_candidate_words(collection, dictionary_collection, model, dictionary, iteration_number):
    query = {
        "relevant": True,
        "iteration_number": iteration_number
    }
    projection = {
        "tokens": 1
    }
    tweets = list(collection.find(query, projection))

    total_annotated_tweets = collection.find().count()
    total_correct_tweets = len(tweets)

    words = set()

    for t in tweets:
        words = words.union(t["tokens"])

    new_words = False
    words = words - set(dictionary)

    bulk = dictionary_collection.initialize_ordered_bulk_op()
    counter = 0

    for w in words:
        print(w)
        if w in model:
            similar_words = model.similar_by_word(w, topn=10)
            similar_words = set(map(lambda x: x[0], similar_words))
            common_words = similar_words.intersection(dictionary)
            if len(common_words) > 0:
                counter += 1
                new_words = True
                score = compute_word_score(w, total_annotated_tweets, total_correct_tweets, collection)

                if score<0.5:
                    continue

                bulk.insert({
                    "word": w,
                    "iteration": iteration_number,
                    "origin": list(common_words),
                    "score": score
                })
                print(w, "added to the dictionary")
                if counter % 10 == 0:
                    print("Executing a bulk insert")
                    bulk.execute()
                    bulk = dictionary_collection.initialize_ordered_bulk_op()

    if counter % 10 != 0:
        print("Executing the rest of the bulk")
        bulk.execute()

    return new_words


def compute_word_score(word, tweets_number, tweets_correct_number, tweets_collection):
    occurrence_query = {
        "tokens": word,
        "relevant": {
            "$exists": True
        }
    }

    occurrence_correct_query = {
        "tokens": word,
        "relevant": True
    }

    occurrence = tweets_collection.find(occurrence_query).count()
    occurrence_correct = tweets_collection.find(occurrence_correct_query).count()

    score = (occurrence_correct / tweets_correct_number) * (tweets_correct_number / tweets_number) / (
                occurrence / tweets_number)
    return score


def evaluate_candidates_tfidf(space, svd, candidates, tweet_seeds):
    seed_vectors = []
    for t in tweet_seeds:
        vector = svd.transform(space.transform([t["tweet_text"]]))
        # vector = vector.todense()
        seed_vectors.append(vector[0])

    seed_vectors = np.average(seed_vectors, axis=0)

    print(seed_vectors)
    similarities = []
    for c in candidates:
        vector = svd.transform(space.transform([c["tweet_text"]]))
        # vector = vector.todense()
        print(vector)
        similarity = cosine_similarity(
            vector[0].reshape(1, -1), seed_vectors.reshape(1, -1))

        similarities.append([similarity[0], c["tweet_text"], c["label"]])

    return sorted(similarities, key=lambda x: -x[0])


def setup():
    print("Configuring the logger")
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]

    print("Loading the models")
    word2vec = models.KeyedVectors.load_word2vec_format(
        "../models/GoogleNews-vectors-negative300.bin", binary=True)

    # word2vec = models.KeyedVectors.load("tweet_model_word2vec.bin")
    print(word2vec)
    doc2vec = models.Doc2Vec.load("../models/tweet_model_doc2vec_v2.bin")
    print(doc2vec)

    dictionary_collection = db["dictionary"]
    twitter_collection = db["tweet"]

    return dictionary_collection, twitter_collection, word2vec, doc2vec


def main(iteration_number=50):
    dictionary_collection, twitter_collection, word2vec, doc2vec = setup()

    for i in range(0, iteration_number):

        print("Starting iteration number ", i)
        dictionary = read_dictionary(dictionary_collection, i - 1)

        seeds = get_seeds(twitter_collection)

        print("Retrieving the candidates")
        candidates = select_candidates_tweet(twitter_collection, dictionary)

        print(len(candidates), " candidates found")

        if len(candidates) == 0:
            print("No more candidates found")
            break

        print("Evaluating the candidates")
        similarities = evaluate_candidate(doc2vec, candidates, seeds, word2vec)

        print("Annotating the candidates")
        new_tweets_number = annotate_candidates(similarities, 0.0, 0.8, twitter_collection, i)

        print("Expanding the dictionary")
        new_words = select_candidate_words(twitter_collection, dictionary_collection, word2vec, dictionary, i)

        if new_tweets_number == 0 and not new_words:
            print("Cannot further annotate any tweets")
            break

    print("Done in ", i, " iterations")


main()
