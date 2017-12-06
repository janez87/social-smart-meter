"""docstring """
import re
import config
import tweepy
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

API_KEY = config.API_KEY
API_SECRET = config.API_SECRET

ACCESS_TOKEN = config.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = config.ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

twitter = tweepy.API(auth)

geocode = '52.3547746,4.758197,10km'  # latitude,longitude,distance(mi/km)

results = tweepy.Cursor(twitter.search, q="*", geocode=geocode).items()

with open("out4.txt", mode="a") as o:
    for result in results:
        if result.lang == "en":
            text = result.text
            text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
            text = text.replace('\n', " ")
            o.write(text+"\n")
            print(result.id_str+","+text)

