#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Importing neccessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[25]:


#Load train and test dataset

train_data = pd.read_csv('fraudTrain.csv', nrows=10000)
test_data = pd.read_csv('fraudTest.csv', nrows=10000)


# In[26]:


train_data.head(10)


# In[27]:


test_data.head(10)


# In[28]:


train_data.shape


# In[29]:


X_train = train_data.drop(['is_fraud','trans_date_trans_time','merchant','category','first','last','street','gender','city','state','job','dob','trans_num'], axis=1)
y_train = train_data['is_fraud']

X_test = test_data.drop(['is_fraud','trans_date_trans_time','merchant','category','first','last','street','gender','city','state','job','dob','trans_num'], axis=1)
y_test = test_data['is_fraud']


# In[30]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[31]:


# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)


# In[32]:


# Evaluate the Logistic Regression model
lr_predictions = lr_model.predict(X_test_scaled)
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)
lr_classification_report = classification_report(y_test, lr_predictions)

print("Logistic Regression Confusion Matrix:\n", lr_confusion_matrix)
print("Logistic Regression Classification Report:\n", lr_classification_report)


# In[33]:


# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)


# In[34]:


# Evaluate the Decision Tree model
dt_predictions = dt_model.predict(X_test_scaled)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
dt_classification_report = classification_report(y_test, dt_predictions)

print("Decision Tree Confusion Matrix:\n", dt_confusion_matrix)
print("Decision Tree Classification Report:\n", dt_classification_report)


# In[35]:


# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)


# In[36]:


# Evaluate the Random Forest model
rf_predictions = rf_model.predict(X_test_scaled)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)

print("Random Forest Confusion Matrix:\n", rf_confusion_matrix)
print("Random Forest Classification Report:\n", rf_classification_report)


# In[39]:


import matplotlib.pyplot as plt
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate models and collect metrics
models = [lr_model, dt_model, rf_model]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

for model in models:
    predictions = model.predict(X_test_scaled)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=1)
    metrics['Accuracy'].append(report['accuracy'])
    metrics['Precision'].append(report['1']['precision'])
    metrics['Recall'].append(report['1']['recall'])
    metrics['F1-Score'].append(report['1']['f1-score'])

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = range(len(model_names))

for i, metric in enumerate(metrics.keys()):
    ax.bar([pos + width * i for pos in x], metrics[metric], width=width, label=metric)

ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels(model_names)
ax.legend()
ax.set_xlabel('Models')
ax.set_title('Model Performance Comparison')
plt.tight_layout()
plt.show()


# In[ ]:




