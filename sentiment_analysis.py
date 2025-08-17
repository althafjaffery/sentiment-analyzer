# Sentiment Analysis using Hugging Face Transformers
# Author: Your Name

from transformers import pipeline

# Load pre-trained sentiment-analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Example texts
texts = [
    "I love learning AI!",
    "This movie was terrible.",
    "The service was okay, nothing special."
]

# Run sentiment analysis
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")
    print("-" * 50)
