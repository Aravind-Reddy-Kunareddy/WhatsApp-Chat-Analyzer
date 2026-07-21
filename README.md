# WhatsApp Chat Analyzer

A Machine Learning and NLP-powered WhatsApp Chat Analyzer built with **Python** and **Streamlit**. Upload your exported WhatsApp chat and gain meaningful insights through interactive visualizations, statistical analysis, and AI-powered predictions.

---

## Features

### Chat Analytics
- Total Messages
- Total Words
- Media Shared
- Links Shared
- Most Active Users (Group Chats)
- Monthly Timeline
- Daily Timeline
- Weekly Activity Map
- Monthly Activity
- Word Cloud
- Most Common Words
- Emoji Analysis

---

### Machine Learning Features

#### User Identification
Predict which participant most likely sent a given message using **TF-IDF** and **Logistic Regression**.

#### Sentiment Analysis
Classify messages into:
- Positive
- Neutral
- Negative

Powered by **Hugging Face Transformers**.

#### Emotion Detection
Identify emotions such as:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

---

### Personality Analysis

Analyze each user's communication style based on:

- Message Frequency
- Average Message Length
- Average Word Count
- Emoji Usage
- Question Usage
- Exclamation Usage

Uses **K-Means Clustering** to group users with similar chatting behavior.

---

### Chat Topic Summary

Instead of generating summaries using Large Language Models, this project discovers the **main discussion topics** using **Topic Modeling**.

Pipeline:
- Text Cleaning
- Stopword Removal
- TF-IDF Vectorization
- K-Means Clustering
- Keyword Extraction
- Topic-wise Representative Messages


## Tech Stack

### Programming Language
- Python

### Frontend
- Streamlit

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Plotly
- Seaborn
- WordCloud

### Machine Learning
- Scikit-learn
- Logistic Regression
- K-Means Clustering
- TF-IDF Vectorizer

### NLP
- NLTK
- Hugging Face Transformers

---

## Project Structure

```
Whatsapp_chat_analyzer/
│
├── app.py
├── helper.py
├── topic_modeling.py
├── requirements.txt
├── README.md
│
├── models/
    ├── preprocessor.py
    ├── sentiment_analysis.py
    ├── emotion_detection.py
    ├── personality_analysis.py
    ├── user_identification.py
```
