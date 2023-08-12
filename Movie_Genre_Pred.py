#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


data=pd.read_csv('train_data.csv')


# In[8]:


data.head(10)


# In[9]:


data.tail(10)


# In[10]:


data.shape


# In[11]:


data.info


# In[12]:


#Check for missing value

print("Any missing value?",data.isnull().values.any())


# In[13]:


data.describe()


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# In[15]:


test=pd.read_csv('test_data.csv')


# In[16]:


test.head()


# In[19]:


train_data = pd.read_csv('train_data.csv', sep=':::', header=None, engine='python')
train_texts = train_data[1]  # Extract movie descriptions
train_labels = train_data[2].str.strip()  # Extract and clean genre labels


# In[20]:


test_data = pd.read_csv('test_data.csv', sep=':::', header=None, engine='python')
test_texts = test_data[1]  # Extract movie descriptions
test_labels = test_data[2].str.strip()  # Extract and clean genre labels


# In[21]:


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)


# In[22]:


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, train_labels)


# In[23]:


predicted_labels = svm_classifier.predict(X_test)


# In[25]:


genre_counts = train_labels.value_counts()
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.title('Distribution of Genres in Training Data')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[26]:


genre_counts_test = pd.Series(predicted_labels).value_counts()
plt.subplot(1, 2, 2)
plt.bar(genre_counts_test.index, genre_counts_test.values, color='orange')
plt.xlabel('Predicted Genre')
plt.ylabel('Number of Movies')
plt.title('Distribution of Predicted Genres in Test Data')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[ ]:




