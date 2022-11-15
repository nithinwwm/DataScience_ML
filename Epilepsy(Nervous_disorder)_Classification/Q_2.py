#!/usr/bin/env python
# coding: utf-8

# ## Q_2 Epilepsy
# 
# Epilepsy is a nervous system disorder that affects movement. The dataset
# contains 195 records of various people with 23 features that contain
# biomedical measurements. Your model will be used to differentiate
# healthy people from people having the disease. Target Column is 'status'.
# Identify the model with the best params. Please note: Visualisation is
# mandatory. You will receive 0 marks if you do not add visualisation. Data
# Link -
# https://github.com/edyoda/data-science-complete-tutorial/blob/master
# /Data/epilepsy.data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#url =  'https://github.com/edyoda/data-science-complete-tutorial/blob/master /Data/epilepsy.data'
df = pd.read_csv('epilepsy.data.csv')


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.describe()


# In[7]:


df.columns


# In[ ]:


## Handling missing values


# In[13]:


for col in df.columns:
    if df[col].isna().sum() > 0:
        print(col, ' : ' ,df[col].isna().sum())


# In[9]:


df[df['MDVP:PPQ'].isnull()]


# In[14]:


df[df['Jitter:DDP'].isnull()]


# In[15]:


df['MDVP:PPQ'].fillna(df['MDVP:PPQ'].mode()[0], inplace=True)
df['Jitter:DDP'].fillna(df['Jitter:DDP'].mode()[0], inplace=True)


# In[16]:


df.isna().any()


# In[ ]:


## Trying to find Co-relation among the features
## Data Visualization


# In[70]:


sns.pairplot(df.iloc[:,:5],kind='scatter')


# In[17]:


ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(df.corr(),cmap='YlGnBu',annot=True, annot_kws = {'size': 6}, fmt='.1g')


# In[64]:


df.corr()['status'].plot(kind='bar')


# In[40]:


## Processing target_column

df['status'].value_counts()


# In[15]:


df.head(1)


# In[17]:


df['name'].isna().


# In[42]:


##Since df.name is of no relation with the target column we can drop df['name']
df.drop('name',axis=1,inplace=True)
df.head()


# # Data Pre-processing
# ## Independent and dependent features
# 

# In[ ]:


X = df.drop('status',axis=1)
y = df['status']


# In[39]:


X.head()


# In[44]:


y.head()


# In[ ]:





# In[ ]:





# ## Train-Test split

# In[45]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)


# In[89]:


print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)


# In[90]:


print('X_test : ',X_test.shape)
print('y_test : ',y_test.shape)


# ## Feature_Scaling

# In[91]:


from sklearn.preprocessing import StandardScaler


# In[93]:


ss = StandardScaler()
ss.fit(X_train)


# In[94]:


X_train = ss.transform(X_train)


# In[95]:


print(X_train)


# In[96]:


X_test = ss.transform(X_test)


# In[97]:


print(X_test)


# ## Model Building
# 1. Logistic Regression
# 2. Random Forest Classifier
# 2. SVM Classifier
# 3. XGB Classifier

# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


lr = LogisticRegression(max_iter=100)


# In[73]:


lr.fit(X_train,y_train)


# In[74]:


X_pred = lr.predict(X_train)


# In[123]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix


# In[76]:


X_accuracy = accuracy_score(y_train,X_pred)
X_accuracy


# In[77]:


y_pred = lr.predict(X_test)


# In[78]:


accuracyscore = accuracy_score(y_pred,y_test)
print('The accuracy score for Logistic Regression Model : ',accuracyscore)


# ##

# In[79]:


from sklearn.ensemble import RandomForestClassifier


# In[81]:


rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred_RFC = rfc.predict(X_test)
accuracyscoreRFC = accuracy_score(y_test,y_pred_RFC)
print('The accuracy score for Random Forest Classifier Model : ',accuracyscoreRFC)


# In[98]:


from sklearn import svm
from sklearn.svm import LinearSVC


# In[99]:


svc = LinearSVC()


# In[100]:


svc.fit(X_train, y_train)


# In[117]:


X_pred_svc = svc.predict(X_train)
accuracyscoreSVC = accuracy_score(y_train,X_pred_svc)


# In[116]:


y_pred_svc = svc.predict(X_test)
accuracy_score(y_pred_svc,y_test)

print('The accuracy score for Support Vector Classifier Model : ',accuracyscoreSVC)


# In[120]:


print(confusion_matrix(y_test,y_pred_svc))


# In[138]:


plot_confusion_matrix(svc, X_test, y_test,cmap='GnBu')


# ##

# In[108]:


from xgboost import XGBClassifier


# In[109]:


model = XGBClassifier(learning_rate=0.1, max_depth=20,verbosity=2,random_state=42,scale_pos_weight=1.5, eval_metric='mlogloss',use_label_encoder =False)


# In[110]:


model.fit(X_train,y_train)


# In[114]:


y_predXGB = model.predict(X_test)


# In[128]:


accuracyscoreXGB = accuracy_score(y_test,y_predXGB)


# In[121]:


print(confusion_matrix(y_test,y_predXGB))


# In[129]:


print('The accuracy score for XGBClassifier Model : ',accuracyscoreXGB)


# In[148]:


plot_confusion_matrix(model, X_test, y_test,cmap='ocean_r')


# In[149]:


print(classification_report(y_test,y_predXGB))


# In[ ]:




