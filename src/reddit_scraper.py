import snscrape.modules.reddit as sreddit
import pandas as pd
import time
import praw
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

def get_reddit_posts(query, limit):
    posts = []
    num_posts = 0
    for post in sreddit.RedditSearchScraper(query).get_items():
        try:
            date = post.date
            text = post.body
            if text == "[removed]" or text == "[deleted]":
                continue
            posts.append([date, text])
            num_posts += 1
        except AttributeError:
            continue
        if num_posts >= limit:
            break

    df = pd.DataFrame(posts, columns=["Date", "Text"])
    print(df)
    return df

def search_reddit(query, limit):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    search_results = reddit.subreddit('all').search(query, limit=limit)

    posts = []
    for post in search_results:
        timestamp = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        posts.append({'timestamp': timestamp, 'content': post.selftext})

    df = pd.DataFrame(posts)

    print(df)

if __name__ == "__main__":  
    time_start = time.time()
    search_reddit("Canadian economy", 10)
    time_end = time.time()
    print(f"\nTotal time taken: {time_end - time_start:.2f} seconds")

