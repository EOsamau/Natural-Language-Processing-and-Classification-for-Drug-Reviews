import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import joblib
from tqdm import tqdm  # For progress bars
import logging         # For logging

# Set up logging to display info in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load DistilBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = AutoModel.from_pretrained('distilbert-base-uncased')

# Function to generate embeddings for a batch of texts
def get_batch_embeddings(texts, max_length=128):
    """
    Generate DistilBERT embeddings for a batch of texts.
    """
    encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        outputs = distilbert_model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Load and preprocess data
logging.info("Loading and preprocessing data...")
train_data = pd.read_csv('Data/drugsComTrain_raw.tsv', sep='\t')
test_data = pd.read_csv('Data/drugsComTest_raw.tsv', sep='\t')
train_data = pd.concat([train_data, test_data], axis=0)

# Convert ratings to sentiment (binary classification)
train_data['sentiment'] = train_data['rating'].apply(lambda rating: 1 if rating > 5 else 0)

# Generate embeddings in batches
logging.info("Generating DistilBERT embeddings...")
embeddings = []
batch_size = 32

# Use tqdm for progress bar in batch processing
for i in tqdm(range(0, len(train_data), batch_size), desc="Processing Batches"):
    batch = train_data['review'].iloc[i:i+batch_size].tolist()
    batch_embeddings = get_batch_embeddings(batch)
    embeddings.extend(batch_embeddings)

X = np.vstack(embeddings)
y = train_data['sentiment']
logging.info("Embeddings generation completed.")

# Split the data
logging.info("Splitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Scale the features
logging.info("Scaling the features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
logging.info("Applying PCA for dimensionality reduction...")
pca = PCA()
cumulative_variance = np.cumsum(pca.fit(X_train_scaled).explained_variance_ratio_)
num_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

# Apply PCA reduction
pca = PCA(n_components=num_components_99)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Train XGBoost classifier
logging.info("Training the XGBoost classifier...")
xgb_classifier = XGBClassifier(
    tree_method='approx',
    n_estimators=100,
    max_depth=25,
    max_delta_step=2,
    learning_rate=0.17,
    scale_pos_weight=2,
    reg_lambda=2,
    reg_alpha=2,
    random_state=42,
    n_jobs=-1
)
xgb_classifier.fit(X_train_reduced, y_train)
logging.info("XGBoost training completed.")

# Print classification report
logging.info("Generating classification report...")
xgb_pred = xgb_classifier.predict(X_test_reduced)
print("\nXGBoost Classifier Results:")
print(classification_report(y_test, xgb_pred, target_names=['Negative', 'Positive']))

# Plot ROC curve
logging.info("Plotting the ROC curve...")
xgb_prob = xgb_classifier.predict_proba(X_test_reduced)
fpr, tpr, _ = roc_curve(y_test, xgb_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('Plots/Broc_curve_distilbert.png')
plt.show()

logging.info("Saving models and preprocessing pipeline...")
# Save models and preprocessing pipeline
torch.save(distilbert_model.state_dict(), "Models/distilbert_model.pt")
joblib.dump(tokenizer, "Models/distilbert_tokenizer.pkl")
joblib.dump(scaler, "Models/scaler_distilbert.pkl")
joblib.dump(pca, "Models/pca_distilbert.pkl")
joblib.dump(xgb_classifier, "Models/model_classifier_distilbert.pkl")
logging.info("All steps completed successfully.")
