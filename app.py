import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# load models
logistic = pickle.load(open("logistic_model.pkl", "rb"))
nb = pickle.load(open("nb_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_model(model, news):
    cleaned = clean_text(news)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed).max() * 100

    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return label, confidence

st.title("Fake News Detection System")
st.write("Compare predictions from multiple ML models")

news = st.text_area("Paste news article here")

if st.button("Predict"):
    if news.strip():
        log_pred, log_conf = predict_model(logistic, news)
        nb_pred, nb_conf = predict_model(nb, news)
        rf_pred, rf_conf = predict_model(rf, news)

        st.subheader("Model Predictions")

        st.write(f"Logistic Regression: {log_pred} ({log_conf:.2f}%)")
        st.write(f"Naive Bayes: {nb_pred} ({nb_conf:.2f}%)")
        st.write(f"Random Forest: {rf_pred} ({rf_conf:.2f}%)")

        best = max(
            [
                ("Logistic Regression", log_conf),
                ("Naive Bayes", nb_conf),
                ("Random Forest", rf_conf)
            ],
            key=lambda x: x[1]
        )

        st.success(f"Best Model For This Input: {best[0]} ({best[1]:.2f}%)")

    else:
        st.warning("Please enter news text")
