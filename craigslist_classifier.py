import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import re
# nltk
from nltk.stem import WordNetLemmatizer

model = joblib.load("model/craigslist_classifier_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

# Function to preprocess text
def preprocess_text(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)      
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText

# Function to predict category
def predict_category(city, section, heading):
    heading = preprocess_text(heading)
    features = vectorizer.transform(heading)
    prediction = model.predict(features)
    return prediction[0]

# Main Streamlit app
def main():
    st.title("Craigslist Post Category Classifier")

    city = st.selectbox("City", ['newyork', 'seattle', 'chicago', 'london', 'manchester', 'hyderabad', 'mumbai', 'delhi', 'singapore', 'bangalore', 'paris.en', 'geneva.en', 'zurich.en','frankfurt.en', 'kolkata.en', 'dubai.en'])
    section = st.selectbox("Section", ['for-sale', 'housing', 'community', 'services'])
    heading = st.text_input("Heading", "")

    if st.button("Predict Category"):
        category = predict_category(city, section, heading)
        st.success(f"Predicted Category: {category}")
        
    # Read JSONL file
    st.sidebar.title("Select a JSONL file")
    uploaded_file = st.sidebar.file_uploader("Upload a JSONL file", type=["json"])

    if uploaded_file is not None:
        df = pd.read_json(uploaded_file, lines=True)
        
        # Predict category for each row in the DataFrame
        df['category'] = df.apply(lambda row: predict_category(row['city'], row['section'], row['heading']), axis=1)

        # Display the results in a table
        st.write("Predicted Categories:")
        st.write(df)
        st.sidebar.write(df) 

if __name__ == "__main__":
    main()
