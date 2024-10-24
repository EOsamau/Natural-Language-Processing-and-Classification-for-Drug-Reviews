# %% [markdown]
# ## Data & EDA

# %% [markdown]
# ### Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE



from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score






# %%
#loading the stopwords library and the nltk to be used for the text preprocessing
nltk.download('punkt')
nltk.download('stopwords')

# %%
train_data = pd.read_csv('drugsComTrain_raw.tsv', sep='\t')
test_data = pd.read_csv('drugsComTest_raw.tsv', sep='\t')

train_data = pd.concat([train_data, test_data], axis=0)
train_data.head()

# %%
train_data.shape

# %%
train_data['condition'].value_counts()

# %%
plt.figure(figsize=(10, 6))
sns.histplot(train_data['rating'], bins=10)
plt.title('Ratings on Drug Reviews by Users')
plt.savefig('histogram_of_ratings.png')
plt.show()

# %%
plt.figure(figsize=(10, 6))
top_10_conditions = train_data['condition'].value_counts().head(10)
sns.barplot(x=top_10_conditions.values, y=top_10_conditions.index, color='grey')
plt.title('Top 10 Conditions in Drug Reviews')
plt.xlabel('Number of Reviews')
plt.savefig('Top_10_conditions.png')
plt.show()


# %%
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens) #--- Joins the token into a simple word. 

train_data['processed_review'] = train_data['review'].apply(preprocess_text)
test_data['processed_review'] = test_data['review'].apply(preprocess_text)
train_data['processed_review'].head()

# %%
#creating a word cloud to visualize the most used words in 215,063 drug reviews by the users
all_words = ' '.join(train_data['processed_review'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.title('Word Cloud of All Reviews')

plt.savefig('wordcloud.png')
plt.show()

# %% [markdown]
# ## Text Processing and Feature Engineering

# %% [markdown]
# #### TF-IDF

# %%
#---------TF-IDF -----------#

#Now in this stage we want to assign importance to words in the training data by using the TF IDF method which 
#can help distinguish words in accordance to how they are most used for ratings

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(train_data['processed_review'])


# %% [markdown]
# ### Word2Vec

# %%

# Training a Word2Vec model tokenized reviews using the SkipGram method
word2vec_model = gensim.models.Word2Vec(sentences=train_data['processed_review'], vector_size=300, window=5, min_count=2, sg=1, workers=4)


def get_average_word2vec(tokens_list, model, vector_size):
    
    valid_words = [word for word in tokens_list if word in model.wv.key_to_index] #vocabulary from word2Vec model
    
    if not valid_words: 
        return np.zeros(vector_size) #we want to return a zero vector for the doc in case no valid words are found in the doc
    
    return np.mean(model.wv[valid_words], axis=0) # the entire review is converted into a single vector, 
                                                  #instead of different vectors for each word in the document

vector_size = word2vec_model.vector_size
train_data['word2vec_review'] = train_data['processed_review'].apply(lambda x: get_average_word2vec(x, word2vec_model, vector_size))


X_word2vec = np.vstack(train_data['word2vec_review'].values) # To convert the into a numpy array


# %% [markdown]
# ### Encoding Ratings to Sentiments

# %%
def categorize_sentiment(rating):
    if rating <= 4:
        return 0  # Negative
    elif rating <= 7:
        return 1  # Neutral
    else:
        return 2  # Positive
    
train_data['sentiment'] = train_data['rating'].apply(categorize_sentiment)
y = train_data['sentiment']

# %% [markdown]
# ##### Trein-Test Split for TF_IDF and Word2Vec Models

# %%


Xtf_train, Xtf_test, ytf_train, ytf_test = train_test_split(X_tfidf, y, test_size=0.15, random_state = 1)
Xw2v_train, Xw2v_test, yw2v_train, yw2v_test = train_test_split(X_word2vec, y, test_size=0.15, random_state = 1)

scaler = MaxAbsScaler()
Xtf_train_scaled = scaler.fit_transform(Xtf_train)
Xtf_test_scaled = scaler.transform(Xtf_test)



# %%
scaler2 = StandardScaler()
Xw2v_train_scaled = scaler2.fit_transform(Xw2v_train)
Xw2v_test_scaled = scaler2.transform(Xw2v_test)

# %%
Xw2v_train_scaled.shape


# %% [markdown]
# ## Model Training and Evaluation

# %% [markdown]
# #### TFIDF - evaluating the XGBoost, Random Forest and Logistic

# %%

svd = TruncatedSVD(n_components=10)
X_train_svd = svd.fit_transform(Xtf_train_scaled)
X_test_svd = svd.transform(Xtf_test_scaled)

# %%

xgb_classifier = XGBClassifier(n_estimators=100, subsample = 0.88,
                               learning_rate=0.1, random_state = 42, n_jobs=-1)  
xgb_classifier.fit(X_train_svd, ytf_train)
xgb_pred = xgb_classifier.predict(X_test_svd)

print("\XGBoost Classifier Results:")
print(classification_report(ytf_test, xgb_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=1))

# %%

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5,
                                           n_jobs=-1, random_state=42)
rf_classifier.fit(X_train_svd, ytf_train)
rf_pred = rf_classifier.predict(X_test_svd)

print("\nRandom Forest Classifier Results:")
print(classification_report(ytf_test, rf_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=1))

# %%

logistic_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
logistic_classifier.fit(X_train_svd, ytf_train)
logistic_pred = logistic_classifier.predict(X_test_svd)

print("\nLogistic Regression Results:")
print(classification_report(ytf_test, logistic_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=1))

# %% [markdown]
# Without wasting compute time with max_depth, it appears the overall accurcay is around 62% when XGBoost and RandomForest are used for the classification. We will now proceed to use the optimal number of principal components to see if the accurcay improves. After several iterations it appears the XGBoost performs better than the Random Forest Classifier. 
# 
# Also, we notice here that there are class imbalances and there we need to adjust the data to ensure there is some balance.

# %% [markdown]
#  ### Word2Vec - Evaluating the Logistic, Random Forest and XGBoost Model 

# %%
pca = PCA(n_components=10)  
X_train_pca = pca.fit_transform(Xw2v_train_scaled)
X_test_pca = pca.transform(Xw2v_test_scaled)

xgb_classifier = XGBClassifier(n_estimators=100, subsample = 0.88,
                               learning_rate=0.1, random_state = 42, n_jobs=-1)  
xgb_classifier.fit(Xw2v_train_scaled, yw2v_train)
xgb_pred = xgb_classifier.predict(Xw2v_test_scaled)

print("\XGBoost Classifier Results:")
print(classification_report(yw2v_test, xgb_pred, target_names=['Negative', 'Neutral', 'Positive']))



# %%

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5,
                                           n_jobs=-1, random_state=42)
rf_classifier.fit(Xw2v_train_scaled, yw2v_train)
rf_pred = rf_classifier.predict(Xw2v_test_scaled)

print("\nRandom Forest Classifier Results:")
print(classification_report(yw2v_test, xgb_pred, target_names=['Negative', 'Neutral', 'Positive']))

# %%

logistic_classifier = LogisticRegression(multi_class='multinomial', solver='saga', random_state=42, max_iter=500)
logistic_classifier.fit(Xw2v_train_scaled, yw2v_train)
logistic_pred = logistic_classifier.predict(Xw2v_test_scaled)

print("\nLogistic Regression Results:")
print(classification_report(yw2v_test, xgb_pred, target_names=['Negative', 'Neutral', 'Positive']))

# %% [markdown]
# ### Reducing Features with PCA assuming Linearity

# %%


pca = PCA().fit(Xw2v_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')

# components that explain 99% of the variance
num_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

plt.axvline(x=num_components_99, color='g', linestyle='--', label=f'{num_components_99} Components')
plt.legend()
plt.show()

plt.savefig('PCA_feature_selection.png')

# Fit PCA with the number of components that explain 99% variance, only on the training data
pca = PCA(n_components=num_components_99)
X_train_reduced = pca.fit_transform(Xw2v_train_scaled)

X_test_reduced = pca.transform(Xw2v_test_scaled)



# %% [markdown]
# ### Reducing features with PCA Assuming Non-linearity

# %% [markdown]
# '''from sklearn.decomposition import KernelPCA
# import matplotlib.pyplot as plt
# import numpy as np
# 
# Apply Kernel PCA with an RBF (Radial Basis Function) kernel
# kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True, n_components=Xw2v_train_scaled.shape[1])
# 
# Fit Kernel PCA on the scaled training data
# X_kpca_train = kpca.fit_transform(Xw2v_train_scaled)
# 
# Compute explained variance using eigenvalues of the Kernel PCA
# lambdas = kpca.lambdas_
# explained_variance_ratio = lambdas / np.sum(lambdas)
# cumulative_variance = np.cumsum(explained_variance_ratio)
# 
# Plot the cumulative explained variance
# plt.plot(cumulative_variance)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Cumulative Explained Variance vs. Number of Components')
# 
# Find the number of components that explain 99% of the variance
# num_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
# 
# dd a vertical line at the point where 99% variance is explained
# plt.axvline(x=num_components_99, color='g', linestyle='--', label=f'{num_components_99} Components')
# 
# Show the plot
# plt.legend()
# plt.show()
# 
# Save the plot
# plt.savefig('KernelPCA_feature_selection.png')
# 
# Apply Kernel PCA with n_components that explains 99% of the variance
# kpca = KernelPCA(kernel='rbf', n_components=num_components_99)
# 
# Transform both training and test data
# X_train_reduced = kpca.fit_transform(Xw2v_train_scaled)
# X_test_reduced = kpca.transform(Xw2v_test_scaled) '''
# 

# %% [markdown]
# ### Dealing with the class imbalances

# %%

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reduced, yw2v_train)

# %%
def plot_class_distribution(y_data, title, subplot, color, class_labels):
    unique, counts = np.unique(y_data, return_counts=True)
    plt.subplot(1, 2, subplot)
    plt.bar(unique, counts, color=color)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(unique, class_labels)  

class_labels = ['Negative', 'Neutral', 'Positive']

plt.figure(figsize=(12, 6))

plot_class_distribution(yw2v_train, 'Class Distribution Before SMOTE', 1, 'grey', class_labels)

plot_class_distribution(y_train_balanced, 'Class Distribution After SMOTE', 2, 'green', class_labels)

plt.tight_layout()
plt.savefig('Smote_Balance_Classes.png')
plt.show()



# %%
X_test_reduced.shape

# %% [markdown]
# ## Tuning Parameters of the best XGB model

# %% [markdown]
# best model so far
# 
# xgb_classifier = XGBClassifier(tree_method = 'hist',n_estimators= 100, max_depth = 25, max_delta_step = 2, 
#                                learning_rate=0.5, reg_lambda= 5, reg_alpha = 2, random_state = 42,  n_jobs=-1) 

# %%
xgb_classifier = XGBClassifier(tree_method = 'hist', n_estimators= 100, max_depth = 25, max_delta_step = 2,
                               learning_rate=0.2, reg_lambda= 5, reg_alpha = 2, random_state = 42,  n_jobs=-1)  
xgb_classifier.fit(X_train_reduced, yw2v_train)
xgb_pred = xgb_classifier.predict(X_test_reduced)

print("\nXGBoost Classifier Results:")
print(classification_report(yw2v_test, xgb_pred, target_names=['Negative', 'Neutral', 'Positive']))

# %%
repr = pd.DataFrame(yw2v_test.value_counts()).reset_index()
repr.columns = ['Class', 'Count']  
repr['Class'] = repr['Class'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})  # Map numeric labels to class names

print(repr)


# %% [markdown]
# ## Model Evaluation

# %%
cm = confusion_matrix(yw2v_test, xgb_pred)

cm_df = pd.DataFrame(cm, index= class_names, columns=class_names)
print("Confusion Matrix:\n", cm_df)


# %%
classes = ['Negative Reviews', 'Neutral Reviews', 'Positive Reviews']

# Convert labels to one-hot encoding for ROC curves
yw2v_test_binarized = label_binarize(yw2v_test, classes=[0, 1, 2])
n_classes = yw2v_test_binarized.shape[1]

xgb_prob = xgb_classifier.predict_proba(X_test_reduced)

# ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(yw2v_test_binarized[:, i], xgb_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

colors = ['red', 'blue', 'green']
for i, color in zip(range(n_classes), colors):
    RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i], estimator_name=classes[i]).plot(ax=axs[i // 2, i % 2], color=color)
    axs[i // 2, i % 2].set_title(f'ROC Curve for {classes[i]}')
    axs[i // 2, i % 2].set_xlabel('False Positive Rate')
    axs[i // 2, i % 2].set_ylabel('True Positive Rate')

# micro-average ROC curve and ROC area
fpr_micro, tpr_micro, _ = roc_curve(yw2v_test_binarized.ravel(), xgb_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

axs[1, 1].plot(fpr_micro, tpr_micro, label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc_micro),
               color='deeppink', linestyle=':', linewidth=4)
axs[1, 1].set_title('Micro-average ROC Curve')
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('ROCs.png')
plt.show()


# %% [markdown]
# These ROC curves indicate that the model has high discriminative power across all classes, with particularly strong performance in distinguishing between classes. The high AUC values (all ≥ 0.93) suggest that the model can effectively separate the classes with a low false positive rate across various classification thresholds.

# %% [markdown]
# ## Cross Validation

# %%
cv_scores = cross_val_score(xgb_classifier, X_train_reduced, yw2v_train, cv=5, scoring='f1_macro')

print("Cross-Validation Scores (5-fold):", cv_scores)
print("Mean Cross-Validation F1 Score:", np.mean(cv_scores))




