from transformers import pipeline


def analyze_sentiment(message):

    sentiment_model = pipeline(
        "sentiment-analysis"
    )

    result = sentiment_model(message)

    return result