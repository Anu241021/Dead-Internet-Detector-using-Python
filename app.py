import streamlit as st
import joblib
import pandas as pd

from scipy.sparse import hstack
from collections import Counter
import string
import re

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def vocabulary_diversity(text):
    words = text.split()

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)

def avg_sentence_length(text):
    sentences = text.split('.')

    sentences = [
        s for s in sentences
        if s.strip()
    ]

    if len(sentences) == 0:
        return 0

    total_words = sum(
        len(s.split())
        for s in sentences
    )

    return total_words / len(sentences)

def repetition_score(text):
    words = text.split()

    if len(words) == 0:
        return 0

    counts = Counter(words)

    repeated = sum(
        c-1
        for c in counts.values()
        if c > 1
    )

    return repeated / len(words)

def punctuation_density(text):
    if len(text) == 0:
        return 0

    punct = sum(
        1
        for c in text
        if c in string.punctuation
    )

    return punct / len(text)

def predict_text(text):

    cleaned = clean_text(text)

    tfidf = vectorizer.transform([cleaned])

    features = pd.DataFrame({
        "diversity":[vocabulary_diversity(cleaned)],
        "sentence_length":[avg_sentence_length(cleaned)],
        "repetition":[repetition_score(cleaned)],
        "punctuation":[punctuation_density(text)]
    })

    scaled_features = scaler.transform(features)

    final_input = hstack([
        tfidf,
        scaled_features
    ])

    prediction = model.predict(final_input)[0]

    probabilities = model.predict_proba(final_input)[0]

    return {
        "prediction": int(prediction),
        "ai_probability": float(probabilities[1]),
        "human_probability": float(probabilities[0]),
        "diversity": float(features["diversity"][0]),
        "sentence_length": float(features["sentence_length"][0]),
        "repetition": float(features["repetition"][0]),
        "punctuation": float(features["punctuation"][0])
    }

st.title("🌐 Dead Internet Detector")

st.write(
    "Analyze text for AI-generated patterns."
)

text = st.text_area(
    "Paste text here",
    height=200
)

if st.button("Analyze"):

    if text.strip():

        result = predict_text(text)

        st.subheader("Verdict")

        st.write(
            f"Prediction: {result['prediction']}"
        )

        st.write(
            f"AI Probability: {result['ai_probability']}%"
        )

        st.write(
            f"Human Probability: {result['human_probability']}%"
        )

        st.subheader("Features")

        st.write(
            f"Vocabulary Diversity: {result['diversity']}"
        )

        st.write(
            f"Average Sentence Length: {result['sentence_length']}"
        )

        st.write(
            f"Repetition Score: {result['repetition']}"
        )

        st.write(
            f"Punctuation Density: {result['punctuation']}"
        )
