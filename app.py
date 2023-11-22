import streamlit as st
import json
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import joblib

st.title("My Auto-Defect-Validator App")


uploaded_file = st.file_uploader("Upload your file", type="json")

if uploaded_file is not None:
    file_contents = uploaded_file.read()
    given_data = json.loads(file_contents)
   
if st.button("Process"):
    status_text = st.text("Processing...")


    data = []
    for i in given_data:
        d = {
            "Description":i["Description"],
            
        }
        data.append(d)
        
    defect_df=pd.DataFrame(data)

    status_text = st.text("Dataframe created...")

    def preprocess_corpus(df):
        ps = PorterStemmer()
        corpus = []
    
        for i in range(0, len(df)):
            review = re.sub('[^a-zA-Z]', ' ', df['Description'][i])
            review = review.lower()
            review = review.split()
    
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
    
        return corpus
    
    status_text = st.text("Preprocessing_data...")
    status_text = st.text("Please wait...")

    corpus = preprocess_corpus(defect_df)

    # Creating the TFIDF model
    tv = TfidfVectorizer(max_features=2500)
    X = tv.fit_transform(corpus).toarray()

    Defect_Validation_Model = joblib.load(r'C:\Users\hp\Desktop\DataSciencePro\NLP\Project\model.pkl')


    status_text = st.text("Predicting...")
    
    valid=Defect_Validation_Model.predict(X)
    prediction_labels = ['valid' if val == 1 else 'invalid' for val in valid]
    defect_df['valid or invalid']=prediction_labels
    
    proba_scores = Defect_Validation_Model.predict_proba(X)
    
    status_text = st.text("Almost Done...")
    # Calculate the confidence score for each prediction
    confidence_scores = proba_scores.max(axis=1)
    
    defect_df['confidence_scores']=confidence_scores
    
    st.dataframe(defect_df)
    

    