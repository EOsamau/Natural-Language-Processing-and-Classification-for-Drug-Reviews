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

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text, max_length=512):
    """
    Generate BERT embeddings for given text
    """
    # Prepare the text inputs
    encoded = tokenizer(text,
                       truncation=True,
                       padding=True,
                       max_length=max_length,
                       return_tensors='pt')
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**encoded)
        # Use the [CLS] token embeddings as the sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings[0]

# Load and preprocess data
train_data = pd.read_csv('Data/drugsComTrain_raw.tsv', sep='\t')
test_data = pd.read_csv('Data/drugsComTest_raw.tsv', sep='\t')
train_data = pd.concat([train_data, test_data], axis=0)

# Convert ratings to sentiment (binary classification)
def categorize_sentiment(rating):
    return 1 if rating > 5 else 0  # Positive if > 5, else Negative

train_data['sentiment'] = train_data['rating'].apply(categorize_sentiment)

# Generate BERT embeddings for reviews
print("Generating BERT embeddings...")
embeddings = []
batch_size = 32  # Process in batches to manage memory

for i in range(0, len(train_data), batch_size):
    batch = train_data['review'].iloc[i:i+batch_size].tolist()
    batch_embeddings = [get_bert_embeddings(text) for text in batch]
    embeddings.extend(batch_embeddings)

X = np.vstack(embeddings)
y = train_data['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA()
cumulative_variance = np.cumsum(pca.fit(X_train_scaled).explained_variance_ratio_)
num_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.axvline(x=num_components_99, color='g', linestyle='--', 
            label=f'{num_components_99} Components')
plt.legend()
plt.savefig('Plots/B04PCA_feature_selection_bert.png')
plt.show()

# Apply PCA reduction
pca = PCA(n_components=num_components_99)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Train XGBoost classifier
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
xgb_pred = xgb_classifier.predict(X_test_reduced)

# Print classification report
print("\nXGBoost Classifier Results:")
print(classification_report(y_test, xgb_pred, target_names=['Negative', 'Positive']))

# Plot ROC curve
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
plt.savefig('Plots/Broc_curve_bert.png')
plt.show()

# Save models and preprocessing pipeline
torch.save(bert_model.state_dict(), "Models/bert_model.pt")
joblib.dump(tokenizer, "Models/bert_tokenizer.pkl")
joblib.dump(scaler, "Models/scaler_bert.pkl")
joblib.dump(pca, "Models/pca_bert.pkl")
joblib.dump(xgb_classifier, "Models/model_classifier_bert.pkl")