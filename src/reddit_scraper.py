import snscrape.modules.reddit as sreddit
import pandas as pd
import time

def get_reddit_posts(query, limit):
    posts = []
    num_posts = 0
    tries = 0
    for post in sreddit.RedditSearchScraper(query).get_items():
        tries += 1
        try:
            date = post.date
            text = post.body
            if text == "[removed]" or text == "[deleted]":
                continue
            posts.append([date, text])
            num_posts += 1
        except AttributeError:
            continue
        if num_posts >= limit or tries >= limit * 10:
            break

    df = pd.DataFrame(posts, columns=["Date", "Text"])
    print(df)
    return df

if __name__ == "__main__":  
    time_start = time.time()
    get_reddit_posts("Canadian economy", 10000)
    time_end = time.time()
    print(f"\nTotal time taken: {time_end - time_start:.2f} seconds")