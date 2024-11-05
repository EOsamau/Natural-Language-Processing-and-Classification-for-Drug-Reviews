import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

torch.manual_seed(1)
np.random.seed(1)

nltk.download('punkt')
nltk.download('stopwords')

class Vocabulary:
    ''' This class is used to build a vocabulary from the text data and convert text data to numericalized format
    designed to create and manage a vocabulary (a mapping of words to indices) for natural language processing tasks,
    with a threshold to control which words are included based on their frequency of occurrence. This ensures that
    only that appear at least 5 times are included in the created vocabulary'''
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text):
        tokenized_text = text.split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def preprocess_text(text):
    ''' This is the function that preprocesses the text data by 
    converting it to lowercase, removing special characters and numbers, and removing stopwords.
    It finally returns the processed text as a string after tokenizing it'''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

class DrugReviewsDataset(Dataset):
    def __init__(self, reviews, sentiments, vocab):
        self.reviews = reviews
        self.sentiments = sentiments
        self.vocab = vocab
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, index):
        review = self.reviews[index]
        sentiment = self.sentiments[index]
        
        # Numericalize the review text
        numericalized_review = self.vocab.numericalize(review)
        
        return torch.tensor(numericalized_review), torch.tensor(sentiment, dtype=torch.float32)

class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # The output dimension is hidden_dim*2 because we're using a bidirectional LSTM
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        
        # Pack the padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concat the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        output = self.fc(self.dropout(hidden))
        return torch.sigmoid(output)

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, n_epochs):
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    for epoch in range(n_epochs):
        train_loss = 0
        valid_loss = 0
        
        # Training
        model.train()
        for batch_idx, (text, lengths, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')):
            text, labels = text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            for text, lengths, labels in valid_loader:
                text, labels = text.to(device), labels.to(device)
                
                predictions = model(text, lengths).squeeze(1)
                loss = criterion(predictions, labels)
                
                valid_loss += loss.item()
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTrain Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return train_losses, valid_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for text, lengths, labels in test_loader:
            text = text.to(device)
            
            pred = model(text, lengths).squeeze(1)
            pred_class = (pred > 0.5).int()
            
            predictions.extend(pred_class.cpu().numpy())
            actual_labels.extend(labels.numpy())
    
    return classification_report(actual_labels, predictions), \
           confusion_matrix(actual_labels, predictions)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('Data/drugsComTrain_raw.tsv', sep='\t')
    
    # Preprocess reviews
    print("Preprocessing text...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Convert ratings to binary sentiment
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x > 5 else 0)
    
    # Create vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(df['processed_review'].values)
    
    # Split the data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['processed_review'].values,
        df['sentiment'].values,
        test_size=0.3,
        random_state=42
    )
    
    valid_texts, test_texts, valid_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42
    )
    
    # Create datasets
    train_dataset = DrugReviewsDataset(train_texts, train_labels, vocab)
    valid_dataset = DrugReviewsDataset(valid_texts, valid_labels, vocab)
    test_dataset = DrugReviewsDataset(test_texts, test_labels, vocab)
    
    # Create data loaders
    def collate_batch(batch):
        # Get labels
        labels = [item[1] for item in batch]
        # Get texts and their lengths
        texts = [item[0] for item in batch]
        lengths = [len(text) for text in texts]
        
        # Pad sequences
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        
        return padded_texts, torch.tensor(lengths), torch.tensor(labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    
    # Initialize model
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    
    model = LSTMSentiment(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Training parameters
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    N_EPOCHS = 10
    
    # Train the model
    print("Training model...")
    train_losses, valid_losses = train_model(
        model, train_loader, valid_loader,
        criterion, optimizer, device, N_EPOCHS
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Evaluate the model
    print("Evaluating model...")
    report, conf_matrix = evaluate_model(model, test_loader, device)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save the model and vocabulary
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_params': {
            'vocab_size': VOCAB_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT
        }
    }, 'lstm_sentiment_model.pt')

if __name__ == "__main__":
    main()