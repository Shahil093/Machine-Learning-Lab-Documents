#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv("Expri_Dataset.csv")


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.values


# In[7]:


df.isnull().sum()


# In[8]:


X=df.iloc[:,:-1]
Y=df["Class"].values


# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[10]:


model = GaussianNB()
model.fit(X_train, Y_train)


# In[11]:


print("Naive Bayes score: ",model.score(X_test, Y_test))


# In[12]:


pred=model.predict(X_test)
print(pred)


# In[13]:


from sklearn.metrics import classification_report,accuracy_score, confusion_matrix


# In[14]:


accuracy_score(Y_test, pred)


# In[15]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# In[16]:


auc_score = roc_auc_score(Y_test, pred)


# In[17]:


fpr, tpr, _ = roc_curve(Y_test, pred)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
# Add labels and title to plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
# Show the plot
plt.show()


# In[ ]:




