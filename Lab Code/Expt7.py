#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# In[3]:


df = pd.read_csv('Expri_Dataset.csv')


# In[9]:


df.keys


# In[10]:


df.head(10)


# In[11]:


from sklearn.preprocessing import MinMaxScaler


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


scaler=StandardScaler()
scaler.fit(df)


# In[14]:


scaled_data= scaler.transform(df)


# In[15]:


scaled_data


# In[16]:


from sklearn.decomposition import PCA


# In[79]:


pca=PCA(n_components=3)


# In[80]:


pca.fit(scaled_data)


# In[81]:


x_pca=pca.transform(scaled_data)


# In[82]:


scaled_data.shape


# In[83]:


x_pca.shape


# In[84]:


scaled_data


# In[85]:


x_pca


# In[86]:


X=x_pca
Y=df["Class"].values


# In[87]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=df['Class'])

# Set the axis labels
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')

# Show the plot
plt.show()


# In[88]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[89]:


model = GaussianNB()
model.fit(X_train, Y_train)


# In[90]:


print("Naive Bayes score: ",model.score(X_test, Y_test))


# In[91]:


pred=model.predict(X_test)
print(pred)


# In[92]:


accuracy_score(Y_test, pred)


# In[93]:


auc_score = roc_auc_score(Y_test, pred)
auc_score


# In[94]:


print(classification_report(Y_test,pred))


# In[95]:


fpr, tpr, _ = roc_curve(Y_test, pred)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
# Add labels and title to plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
# Show the plot
plt.show()

