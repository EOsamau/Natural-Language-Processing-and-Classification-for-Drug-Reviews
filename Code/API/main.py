from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = FastAPI()

# Saved models and preprocessors
scaler = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/scaler.pkl")
pca = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/pca.pkl")
xgb_classifier = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/model_classifier.pkl")
word2vec_model = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/Word2Vec.model")
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Define data model for incoming requests
class ReviewData(BaseModel):
    review: list[str]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return tokens

def get_average_word2vec(tokens_list, model, vector_size=300):
    valid_words = [word for word in tokens_list if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_words], axis=0)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = FastAPI()

# Saved models and preprocessors
scaler = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/scaler.pkl")
pca = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/pca.pkl")
xgb_classifier = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/model_classifier.pkl")
word2vec_model = joblib.load("/Users/eosamau/Documents/GitHub/Natural Language Processing and Classification for Drug Reviews/Natural-Language-Processing-and-Classification-for-Drug-Reviews/Code/Models/Word2Vec.model")
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Define data model for incoming requests
class ReviewData(BaseModel):
    reviews: list[str]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return tokens

def get_average_word2vec(tokens_list, model, vector_size=300):
    valid_words = [word for word in tokens_list if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_words], axis=0)

@app.post("/predict")
async def predict_sentiment(data: ReviewData):
    sentiments = []
    for review in data.reviews:
        tokens = preprocess_text(review)
        vector = get_average_word2vec(tokens, word2vec_model)

    # Reshape and scale the data
        vector_scaled = scaler.transform([vector])
        vector_reduced = pca.transform(vector_scaled)

        # Predict sentiment
        prediction = xgb_classifier.predict(vector_reduced)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        sentiments.append(sentiment)

    return {"sentiments": sentiments}

