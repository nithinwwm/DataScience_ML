#!/usr/bin/env python
# coding: utf-8

# ## Q_3
# Predicting Shopping Mall Sales. You will have to create a model to predict
# revenue. Identify the model with the best params. Target Column -
# Revenue. Please note: Visualisation is mandatory.
# #url = 'https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/Shopping_Revenue.csv'

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#url = 'https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/Shopping_Revenue.csv'
df = pd.read_csv('Shopping_Revenue.csv')
df


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.columns


# ## EDA
# * Handling missing values
# * processing categorical columns

# In[35]:


[col for col in df.columns if df[col].isnull().sum()>0]


# In[37]:


df[df['P6'].isnull()]


# In[38]:


df[df['P7'].isnull()]


# In[42]:


df['P6'] = df['P6'].fillna(df['P6'].mode()[0])


# In[47]:


df['P7'] = df['P7'].fillna(df['P7'].mode()[0])


# In[58]:


df.describe()


# In[56]:


ax = plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='crest')


# In[63]:


df_orig = df.copy()


# In[301]:


df.corr()


# In[227]:


plt.figure(figsize = (8,6))
l = list(df_orig['City'].unique())
chart = sns.countplot(df_orig['City'])
chart.set_xticklabels(labels=l,rotation=90)


# In[230]:


plt.figure(figsize = (10,6))
sns.countplot(df['Open_Year'])


# In[66]:


df['Open_Day'] = df['Open Date'].str.split('/').str[1]
df['Open_Month'] = df['Open Date'].str.split('/').str[0]
df['Open_Year'] = df['Open Date'].str.split('/').str[2]


# In[67]:


df.head(1)


# In[68]:


df.drop('Open Date',axis=1, inplace=True)


# In[69]:


df.head(1)


# In[70]:


df['City'].unique()


# In[79]:


df['Open_Year'].unique()


# In[27]:


df['Type'].unique()


# In[138]:


city_grpby = df.groupby(['Open_Year','City','Type','revenue']).size().reset_index()
city_grpby


# In[234]:


ax = plt.figure(figsize=(6,4))
ax = sns.displot(df_orig['revenue'],aspect=1.5,kind='kde',color='green')


# In[223]:


sns.displot(df_orig['revenue'])


# In[155]:


df.head()


# In[152]:


##Processing  Categorical Features


# In[154]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[156]:


df['City'] = le.fit_transform(df['City'])


# In[161]:


df['City Group'] = df['City Group'].map({'Big Cities':1,'Other':0})


# In[165]:


df['Type'] = df['Type'].map({'FC':0,'IL':1,'DT':2})


# In[171]:


df['Open_Day'] = df['Open_Day'].astype(int)
df['Open_Month'] = df['Open_Month'].astype(int)
df['Open_Year'] = df['Open_Year'].astype(int)


# In[172]:


df.head()


# In[235]:


#for i in df.columns:
#    if (len(df.loc[df[i] == 0])) > 5: 
#        print( i ,' : ', (len(df.loc[df[i] == 0])),end=' // ')


# ## Test_train_split

# In[245]:


X = df.drop('revenue',axis=1)
y = df['revenue']


# In[247]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)


# In[248]:


print(X_train.shape , y_train.shape)
print(X_test.shape , y_test.shape)


# In[ ]:





# In[ ]:





# ## Model Building

# In[294]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:





# In[256]:


xgb = XGBRegressor()
xgb.fit(X_train,y_train)


# In[277]:


tra_pred = xgb.predict(X_train)


# In[280]:


r2_train = r2_score(y_train,tra_pred)
r2_train


# In[281]:


test_predict = xgb.predict(X_test)


# In[283]:


r2_test = r2_score(y_test,test_predict)


# In[ ]:





# In[288]:


from sklearn.pipeline import make_pipeline


# In[287]:


def train(model, X, y):
    
    model.fit(X, y)
    pred = model.predict(X)
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:", cv_score)


# In[292]:


model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
train(model, X, y)
#coef = pd.Series(model.coef, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[298]:


model = make_pipeline(StandardScaler(with_mean=False), Ridge())
train(model, X, y)
coef.plot(kind='bar', title="Model Coefficients",color='green')


# In[300]:


model = make_pipeline(StandardScaler(with_mean=False), Lasso())
train(model, X, y)
coef.plot(kind='bar',color='green')


# In[295]:


model = make_pipeline(StandardScaler(with_mean=False), RandomForestRegressor())
train(model, X, y)
coef.plot(kind='bar')


# In[ ]:




