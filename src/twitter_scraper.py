import snscrape.modules.twitter as sntwitter
import pandas as pd

def get_tweets(query, limit):
    tweets = []
    num_tweets = 0
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.rawContent])
        num_tweets += 1
        if num_tweets >= limit:
            break

    df = pd.DataFrame(tweets, columns=["Date", "Text"])
    return df