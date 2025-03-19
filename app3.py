import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Sentiment Mapping
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit App UI
st.title("📢 Social Media Sentiment Analysis")
st.write("Enter comments below or upload a CSV file to analyze their sentiment.")

# User Input (Manual Entry)
comments = st.text_area("Enter comments (one per line):")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def analyze_comments(comment_list):
    comment_vectors = vectorizer.transform(comment_list)
    predictions = model.predict(comment_vectors)
    sentiment_results = [sentiment_labels[p] for p in predictions]
    return sentiment_results

if st.button("Analyze Sentiment"):
    all_comments = []
    
    # Process manually entered comments
    if comments.strip():
        manual_comments = comments.split("\n")
        manual_comments = [c.strip() for c in manual_comments if c.strip()]
        all_comments.extend(manual_comments)
    
    # Process uploaded CSV file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "comments" in df.columns:
            csv_comments = df["comments"].dropna().tolist()
            all_comments.extend(csv_comments)
        else:
            st.error("CSV file must have a column named 'comments'.")
    
    if all_comments:
        sentiments = analyze_comments(all_comments)
        
        # Display results
        st.subheader("Results:")
        for comment, sentiment in zip(all_comments, sentiments):
            st.write(f"**{comment}** → {sentiment}")
        
        # Save results to CSV
        result_df = pd.DataFrame({"Comment": all_comments, "Sentiment": sentiments})
        result_csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", result_csv, "sentiment_results.csv", "text/csv")
    else:
        st.warning("Please enter comments manually or upload a CSV file.")
