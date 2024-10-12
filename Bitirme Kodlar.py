#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import pandas as pd
import re
import tldextract
import numpy as np

# verilerin önişlenmesi ve feature çıkarılması
def preprocess_url(url):
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)

    
    url_length = len(url)
    special_char_count = len(re.findall(r'\W', url))
    http_presence = 1 if parsed_url.scheme == 'http' else 0
    digit_count = sum(c.isdigit() for c in url)
    subdomain_count = len(domain_info.subdomain.split('.')) - 1 if domain_info.subdomain else 0
    common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'uk', 'de', 'jp', 'fr', 'au', 'us', 'ru', 'ch', 'it', 'nl', 'se', 'no', 'es', 'mil']
    tld_common = 1 if domain_info.suffix in common_tlds else 0
    https_presence = 1 if parsed_url.scheme == 'https' else 0
    path_length = len(parsed_url.path)

    return [url_length, special_char_count, http_presence, digit_count, subdomain_count, tld_common, https_presence, path_length]

# veri setinin yüklenmesi
file_path = 'C:/Users/oguzh/Downloads/archive-8/dataset.csv'
data = pd.read_csv(file_path)

# 
features = data['url'].apply(preprocess_url)
features_df = pd.DataFrame(features.tolist(), columns=['url_length', 'special_char_count', 'http_presence', 'digit_count', 'subdomain_count', 'tld_common', 'https_presence', 'path_length'])


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['type'])

# veri setinin eğitim ve test için bölünmesi
X_train, X_test, y_train, y_test = train_test_split(features_df, encoded_labels, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lojistik regresyon modeli
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_predictions = log_reg.predict(X_test_scaled)

# random forest modeli
rand_forest = RandomForestClassifier()
rand_forest.fit(X_train_scaled, y_train)
rand_forest_predictions = rand_forest.predict(X_test_scaled)

# k nearest neighbor modeli
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_predictions = knn.predict(X_test_scaled)


results = {
    "Model": ["Logistic Regression", "Random Forest", "k-Nearest Neighbors"],
    "Accuracy": [accuracy_score(y_test, log_reg_predictions), accuracy_score(y_test, rand_forest_predictions), accuracy_score(y_test, knn_predictions)],
    "Precision": [precision_score(y_test, log_reg_predictions), precision_score(y_test, rand_forest_predictions), precision_score(y_test, knn_predictions)],
    "Recall": [recall_score(y_test, log_reg_predictions), recall_score(y_test, rand_forest_predictions), recall_score(y_test, knn_predictions)],
    "F1 Score": [f1_score(y_test, log_reg_predictions), f1_score(y_test, rand_forest_predictions), f1_score(y_test, knn_predictions)]
}

results_df = pd.DataFrame(results)
print(results_df)


# In[26]:


def classify_url(url, log_reg_model, rand_forest_model, knn_model, scaler, label_encoder):
    
    features = preprocess_url(url)
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    log_reg_prediction = log_reg_model.predict(scaled_features)
    rand_forest_prediction = rand_forest_model.predict(scaled_features)
    knn_prediction = knn_model.predict(scaled_features)  
    log_reg_result = label_encoder.inverse_transform(log_reg_prediction)[0]
    rand_forest_result = label_encoder.inverse_transform(rand_forest_prediction)[0]
    knn_result = label_encoder.inverse_transform(knn_prediction)[0]  

    print(f"URL: {url}")
    print(f"Logistic Regression Prediction: {log_reg_result}")
    print(f"Random Forest Prediction: {rand_forest_result}")
    print(f"k-Nearest Neighbors Prediction: {knn_result}") 

# Siniflandirilacak URL
example_url = "https://datasciencedojo.com/blog/machine-learning-101/#"
classify_url(example_url, log_reg, rand_forest, knn, scaler, label_encoder)

