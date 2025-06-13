import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:/Users/BW/Desktop/DL movie review classifier/outputs/trained_models/bilstm_trained_model.keras")
    return model

# Load the saved tokenizer
@st.cache_resource
def load_tokenizer():
    with open("C:/Users/BW/Desktop/DL movie review classifier/outputs/preprocessing_artifacts/tokenizer.joblib", "rb") as handle:
        tokenizer = joblib.load(handle)
    return tokenizer

# Initialize the model, tokenizer, threshold, and max length
model = load_model()
tokenizer = load_tokenizer()
threshold = 0.7051022052764893
max_length = 200

# App layout
st.title("🎥 Movie Review Sentiment Classifier")
st.subheader("Classify movie reviews as **Positive** or **Negative** with their probabilities.")

# Text input for user reviews
user_review = st.text_area("Enter a Movie Review:", height=90)

# Button for prediction
if st.button("Predict"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess the input review
        review_seq = tokenizer.texts_to_sequences([user_review])
        padded_review = pad_sequences(review_seq, padding='post', truncating='post', maxlen=max_length)  # Fixed truncating typo

        # Get the model's probability prediction
        proba = model.predict(padded_review)[0][0]

        # Apply the threshold to classify as positive or negative
        if proba >= threshold:
            st.success(f"Prediction: Positive")
            st.write(f"Probability that the review is positive: {proba:.2f}")
        else:
            st.error(f"Prediction: Negative")
            st.write(f"Probability that the review is negative: {proba:.2f}")