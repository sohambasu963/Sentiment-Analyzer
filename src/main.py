from twitter_scraper import get_tweets
from reddit_scraper import get_reddit_posts
from text_processing import clean_text, calculate_polarity_score, compound_score, classify_sentiment
import pandas as pd
from transformers import AutoTokenizer
from transformers import TFRobertaForSequenceClassification
import time
    

def main():
    KEYWORD = "Canadian economy"
    LIMIT = 3
    LANG = "en"
    TWITTER_QUERY = f"{KEYWORD} lang:{LANG}"
    REDDIT_QUERY = KEYWORD
    PATH_DIR = "data/tweet_sentiment.csv"

    start_time = time.time()
    # df = get_tweets(TWITTER_QUERY, LIMIT)
    df = get_reddit_posts(REDDIT_QUERY, LIMIT)
    print(df)
    print(len(df))
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
    df['Text'] = df['Text'].apply(clean_text)

    # define model and calculate polarity scores
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL)
    df[['Negative', 'Neutral', 'Positive']] = df['Text'].apply(lambda x: calculate_polarity_score(tokenizer, model, x)).apply(pd.Series)

    # Calculate compound score and sentiment
    df['Compound'] = df.apply(compound_score, axis=1)
    df['Sentiment'] = df['Compound'].apply(classify_sentiment)

    # Save the dataframe to a CSV file
    df.to_csv(PATH_DIR, index=False)

    # Calculate average data about user sentiment
    avg_sentiment = df['Compound'].mean()
    avg_sentiment_classification = classify_sentiment(avg_sentiment)

    print(f"Average sentiment: {avg_sentiment}")
    print(f"Average sentiment classification: {avg_sentiment_classification}")

    # Calculate the percentage of each sentiment category
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    print("\nPercentage of each sentiment category:")
    print(sentiment_counts)
    

if __name__ == "__main__":
    # start_time = time.time()
    main()
    # end_time = time.time()
    # print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
