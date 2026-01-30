import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Mail Detection App")

# ---- Load Dataset ----
@st.cache_data
def load_and_train():
    data = pd.read_csv("emails.csv")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["text"])
    y = data["spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, vectorizer, accuracy

model, vectorizer, accuracy = load_and_train()

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# ---- Prediction ----
email = st.text_area("Enter Email Text")

if st.button("Check Spam"):
    if email.strip() == "":
        st.warning("Please enter some text")
    else:
        email_vec = vectorizer.transform([email])
        pred = model.predict(email_vec)[0]

        if pred == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT Spam")