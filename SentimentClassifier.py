import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import nltk
import joblib

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SentimentClassificationPipeline:
    def __init__(self, word2vec_model, scaler, pca, classifier):
        self.word2vec_model = word2vec_model
        self.scaler = scaler
        self.pca = pca
        self.classifier = classifier

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        return ' '.join(tokens)

    def get_average_word2vec(self, tokens_list, vector_size):
        valid_words = [word for word in tokens_list if word in self.word2vec_model.wv.key_to_index]
        if not valid_words:
            return np.zeros(vector_size)
        return np.mean(self.word2vec_model.wv[valid_words], axis=0)

    def classify(self, sentence):
        # Preprocess the input sentence
        processed_sentence = self.preprocess_text(sentence)
        
        # Convert to Word2Vec representation
        vector_size = self.word2vec_model.vector_size
        word2vec_vector = self.get_average_word2vec(processed_sentence.split(), vector_size)
        
        # Scale the vector
        scaled_vector = self.scaler.transform(word2vec_vector.reshape(1, -1))
        
        # Apply PCA
        reduced_vector = self.pca.transform(scaled_vector)
        
        # Predict using the classifier
        prediction = self.classifier.predict(reduced_vector)
        
        # Map prediction to sentiment label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return sentiment_map[prediction[0]]

# Example usage:
if __name__ == "__main__":
    # Load your pre-trained models and components here
    word2vec_model = gensim.models.Word2Vec.load("word2vec_model.model")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    classifier = joblib.load("xgb_classifier.pkl")

    # Initialize the pipeline
    pipeline = SentimentClassificationPipeline(word2vec_model, scaler, pca, classifier)

    # Example classification
    while True:
        sentence = input("Enter a sentence to classify (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        sentiment = pipeline.classify(sentence)
        print(f"Sentiment: {sentiment}")