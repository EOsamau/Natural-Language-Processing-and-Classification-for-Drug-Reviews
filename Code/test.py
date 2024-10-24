import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim

# Load the saved models
xgb_classifier = joblib.load("xgb_classifier.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
word2vec_model = gensim.models.Word2Vec.load("word2vec_model.model")

# Load stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

def get_average_word2vec(tokens_list, model, vector_size):
    valid_words = [word for word in tokens_list if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_words], axis=0)

def classify_review(review):
    # Preprocess the input review
    processed_review = preprocess_text(review)
    
    # Convert the processed review to Word2Vec vector
    vector_size = word2vec_model.vector_size
    word2vec_vector = get_average_word2vec(processed_review.split(), word2vec_model, vector_size)
    
    # Scale the vector
    scaled_vector = scaler.transform([word2vec_vector])
    
    # Reduce dimensionality with PCA
    reduced_vector = pca.transform(scaled_vector)
    
    # Make prediction using the XGBoost model
    prediction = xgb_classifier.predict(reduced_vector)
    
    return "Negative" if prediction[0] == 0 else "Positive"

# Example of how to use the classify_review function
if __name__ == "__main__":
    new_review = input("Enter a drug review: ")
    sentiment = classify_review(new_review)
    print(f"The predicted sentiment is: {sentiment}")
