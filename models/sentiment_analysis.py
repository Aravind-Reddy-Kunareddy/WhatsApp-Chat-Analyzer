import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

sentiment_model = load_sentiment_model()

def analyze_sentiment(message):
    return sentiment_model(
        message,
        truncation=True,
        max_length=512
    )
