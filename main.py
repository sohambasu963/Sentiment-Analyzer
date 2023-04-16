import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
from transformers import AutoTokenizer
from transformers import TFRobertaForSequenceClassification
from scipy.special import softmax
import time


def calculate_polarity_score(tokenizer, model, tweet):
    encoded_text = tokenizer(tweet, return_tensors='tf')
    output = model(encoded_text)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative' : scores[0],
        'neutral' : scores[1],
        'positive' : scores[2]
    }
    return scores_dict

def preprocess_text(text):
    text = re.sub(r'@(\w)+', '@user', text)
    text = re.sub(r'http\S+', 'http', text)
    return text

def compound_score(row):
    return row['Positive'] - row['Negative']

def classify_sentiment(compound):
    if compound >= 0.7:
        return "Very Positive"
    elif compound >= 0.3:
        return "Positive"
    elif compound > -0.3:
        return "Neutral"
    elif compound > -0.7:
        return "Negative"
    else:
        return "Very Negative"
    
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

def main():
    KEYWORD = "economy"
    LIMIT = 1000
    LANG = "en"
    QUERY = f"{KEYWORD} lang:{LANG}"

    df = get_tweets(QUERY, LIMIT)
    df['Text'] = df['Text'].apply(preprocess_text)

    # define model and calculate polarity scores
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL)
    df[['Negative', 'Neutral', 'Positive']] = df['Text'].apply(lambda x: calculate_polarity_score(tokenizer, model, x)).apply(pd.Series)

    # Calculate compound score and sentiment
    df['Compound'] = df.apply(compound_score, axis=1)
    df['Sentiment'] = df['Compound'].apply(classify_sentiment)

    # Save the dataframe to a CSV file
    df.to_csv('tweets.csv', index=False)

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
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
