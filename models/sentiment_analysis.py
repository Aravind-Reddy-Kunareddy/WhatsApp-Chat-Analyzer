import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

def analyze_sentiment(message):
    result = sentiment_model(
        message,
        truncation=True,
        max_length=512
    )
    return result
