import re
from scipy.special import softmax

def clean_text(text):
    text = re.sub(r'@(\w)+', '@user', text)
    text = re.sub(r'http\S+', 'http', text)
    return text

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