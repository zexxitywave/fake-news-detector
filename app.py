import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# load files
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = ENGLISH_STOP_WORDS

MODEL_NAME = "Logistic Regression"
MODEL_ACCURACY = "98%+"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_news(news):
    cleaned = clean_text(news)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)

    if prediction[0] == 0:
        return "FAKE NEWS"
    return "REAL NEWS"

st.title("Fake News Detection System")

st.write(f"Model Used: {MODEL_NAME}")
st.write(f"Model Accuracy: {MODEL_ACCURACY}")

news = st.text_area("Paste news article here")

if st.button("Predict"):
    if news.strip():
        result = predict_news(news)
        st.success("Prediction: " + result)
    else:
        st.warning("Please enter news text")
