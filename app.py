import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# load saved files
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = ENGLISH_STOP_WORDS


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
    else:
        return "REAL NEWS"


st.title("Fake News Detection System")

st.write("Enter a news article to check whether it is fake or real.")

news = st.text_area("Paste news article here")

if st.button("Predict"):
    if news.strip():
        result = predict_news(news)
        st.success("Prediction: " + result)
    else:
        st.warning("Please enter news text")