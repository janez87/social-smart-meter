#system modules
import sys

#external modules
import tweepy
from pymongo import MongoClient

# my modules
sys.path.append("../")
import config

BOUNDING_BOX = [4.736851,52.273948,5.065755,52.430806]

class StreamCrawler(tweepy.StreamListener):
    
    def __init__(self,collection):
        tweepy.StreamListener.__init__(self)
        self.collection = collection

    def on_status(self,tweet):
        print(tweet.text)
        self.save_tweets(tweet)

    def on_error(self,status_code):
        print(status_code)
        return True
    
    def save_tweets(self,tweet):
        ''' to_save = {
           "json": tweet._json,
            "text": tweet.text,
            "created_at":tweet.created_at,
            "place":tweet.place,
            "id":tweet.id,
            "author":tweet.author
        } '''
        self.collection.insert(tweet._json)

def main():
    print("Connecting to Mongo")
    client = MongoClient(config.DB_HOST, config.DB_PORT)
    db = client[config.DB_NAME]
    twitterCollection = db["tweet_ams"]

    print("Authenticating")
    auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
    auth.set_access_token(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    print("Creating the stream")
    stream_listener = StreamCrawler(twitterCollection)
    stream = tweepy.Stream(auth=api.auth,listener=stream_listener)

    print("Starting the stream")
    stream.filter(locations=BOUNDING_BOX, async=True)

if __name__ == "__main__":
    main()
