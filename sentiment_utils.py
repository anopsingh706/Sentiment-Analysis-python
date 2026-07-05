"""Sentiment scoring helpers."""

from textblob import TextBlob


def analyze_text(message: str) -> dict[str, float | str]:
    sentiment = TextBlob(message).sentiment
    polarity = float(sentiment.polarity)
    subjectivity = float(sentiment.subjectivity)

    if polarity < 0:
        label = "Negative"
    elif polarity > 0:
        label = "Positive"
    else:
        label = "Neutral"

    return {
        "label": label,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }
