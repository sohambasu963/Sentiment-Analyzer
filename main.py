import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoTokenizer
from transformers import TFRobertaForSequenceClassification
from scipy.special import softmax


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

def main():
    keyword = "recession"
    limit = 1000
    query = f"{keyword} lang:en"

    tweets = []
    num_tweets = 0
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.rawContent])
        num_tweets += 1
        if num_tweets >= limit:
            break

    df = pd.DataFrame(tweets, columns=["Date", "Text"])

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL)

    df[['Negative', 'Neutral', 'Positive']] = df['Text'].apply(lambda x: calculate_polarity_score(tokenizer, model, x)).apply(pd.Series)

    # Calculate compound score
    df['Compound'] = df.apply(compound_score, axis=1)

    # Classify sentiment
    df['Sentiment'] = df['Compound'].apply(classify_sentiment)

    # Save the dataframe to a CSV file
    df.to_csv('tweets.csv', index=False)

    # Calculate the average sentiment
    avg_sentiment = df['Compound'].mean()
    avg_sentiment_classification = classify_sentiment(avg_sentiment)

    print(f"Average sentiment: {avg_sentiment}")
    print(f"Average sentiment classification: {avg_sentiment_classification}")

if __name__ == "__main__":
    main()
