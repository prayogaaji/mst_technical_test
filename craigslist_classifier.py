import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load the trained model
model = joblib.load("craigslist_classifier_model.joblib")

# Function to preprocess text
def preprocess_text(text):
    # You can add your preprocessing steps here, such as lowercasing, removing punctuation, etc.
    return text

# Function to predict category
def predict_category(city, section, heading):
    # Preprocess the input text
    heading = preprocess_text(heading)
    # Generate features
    features = vectorizer.transform([heading])
    # Predict category
    prediction = model.predict(features)
    return prediction[0]

# Main Streamlit app
def main():
    st.title("Craigslist Post Category Classifier")

    # Input fields
    city = st.selectbox("City", ["City1", "City2", "City3"])  # Add your city options
    section = st.selectbox("Section", ["For Sale", "Housing", "Community", "Services"])
    heading = st.text_input("Heading", "")

    # Predict category when the button is clicked
    if st.button("Predict Category"):
        category = predict_category(city, section, heading)
        st.success(f"Predicted Category: {category}")

if __name__ == "__main__":
    main()
