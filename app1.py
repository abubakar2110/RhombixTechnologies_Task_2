import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Sentiment Mapping
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit App UI
st.title("📢 Social Media Sentiment Analysis")
st.write("Enter comments below to analyze their sentiment.")

# User Input
comments = st.text_area("Enter comments (one per line):")

if st.button("Analyze Sentiment"):
    if comments.strip():
        comment_list = comments.split("\n")  # Split input into a list
        comment_list = [c.strip() for c in comment_list if c.strip()]  # Remove empty lines
        
        # Transform comments using vectorizer
        comment_vectors = vectorizer.transform(comment_list)
        
        # Predict sentiments
        predictions = model.predict(comment_vectors)
        
        # Convert numerical predictions to labels
        sentiment_results = [sentiment_labels[p] for p in predictions]
        
        # Display results
        st.subheader("Results:")
        for comment, sentiment in zip(comment_list, sentiment_results):
            st.write(f"**{comment}** → {sentiment}")
    else:
        st.warning("Please enter at least one comment.")
