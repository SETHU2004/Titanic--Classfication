#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 

# In[5]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # The Data

# In[6]:


train = pd.read_csv('/home/sethukarasi/Downloads/titanic_train.csv')


# In[7]:


train.head()


# # Exploratory Data Analysis

# # Missing Data

# In[8]:


train.isnull()


# In[9]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[13]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[14]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[15]:


sns.countplot(x='SibSp',data=train)


# In[16]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Cufflinks for plots

# In[17]:


import cufflinks as cf
cf.go_offline()


# In[18]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# # Data Cleaning

# In[19]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[20]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass ==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
        
    else:
        return Age


# In[21]:


train['Age']= train[['Age','Pclass']].apply(impute_age,axis=1)


# In[22]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[23]:


train.drop('Cabin',axis=1,inplace=True)


# In[24]:


train.head()


# In[25]:


train.dropna(inplace=True)


# # Converting Categorical Features

# In[26]:


train.info()


# In[27]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[28]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[29]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[30]:


train.head()


# In[31]:


train = pd.concat([train,sex,embark],axis=1)


# In[32]:


train.head()


# # Building a Logistic Regression model

# # Train Test Split

# In[33]:


train.drop('Survived',axis=1).head()


# In[34]:


train['Survived'].head()


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, Y_train,Y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30,random_state=101)


# # Training and Predicting

# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[39]:


predictions = logmodel.predict(X_test)


# In[40]:


from sklearn.metrics import confusion_matrix


# In[41]:


accuracy=confusion_matrix(Y_test,predictions)


# In[42]:


accuracy


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


accuracy=accuracy_score(Y_test,predictions)
accuracy


# In[45]:


predictions


# # Evaluation
# 

# In[48]:


from sklearn.metrics import classification_report


# In[50]:


print(classification_report(Y_test,predictions))


# In[ ]:




