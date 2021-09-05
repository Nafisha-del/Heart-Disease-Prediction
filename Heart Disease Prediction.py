#!/usr/bin/env python
# coding: utf-8

# Importing Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# The Dataset

# In[2]:


df = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['target'].value_counts()


# In[8]:


df.duplicated()


# In[9]:


df.isnull().sum()


# In[10]:


plt.hist(df)


# In[11]:


sns.boxplot(data=df)


# In[12]:


df.corr()


# In[13]:


plt.figure(figsize=[10,6])
sns.heatmap(df.corr(),annot=True)


# In[14]:


x = df.drop(columns='target', axis=1)
y = df['target']


# In[15]:


print(x)
print(y)


# Splitting into train and test sets

# In[16]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


# In[17]:


print(x.shape, xtrain.shape, xtest.shape)


# **Logistic Regression**

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model = LogisticRegression(solver='liblinear')


# In[20]:


model.fit(xtrain,ytrain)


# In[21]:


ypred = model.predict(xtest)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Decision Tree**

# In[23]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


# In[24]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Random Forest**

# In[25]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)


# In[26]:


ypred=model.predict(xtest)


# In[27]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Support Vector Machine**

# In[28]:


from sklearn.svm import SVC
model=SVC()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


# In[29]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Gaussian Naive Bayes**

# In[30]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


# In[31]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Multinomial Naive Bayes**

# In[32]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


# In[33]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# **Gradient Boosting Classifier**

# In[34]:


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


# In[35]:


cm=confusion_matrix(ytest,ypred)
sns.heatmap(cm,annot=True)
print("accuracy is:",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))


# In[36]:


model = ["LogisticRegression", "Decision Tree", "Random Forest", "Support Vector Machine", "Gaussian Naive Bayes", "Multinomial Naive Bayes", "GradientBoostingClassifier"]
accuracy = [0.8032786885245902*100, 0.7868852459016393*100, 0.7540983606557377*100, 0.6229508196721312*100, 0.819672131147541*100, 0.6885245901639344*100, 0.7377049180327869*100]


# In[37]:


pd.DataFrame({"model":model,"Accuracy(%)":accuracy})


# # We get the highest accuracy score using Gaussian Naive Bayes
