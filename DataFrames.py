#!/usr/bin/env python
# coding: utf-8

# In[185]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split as tts
columns = ['ID','Agency','Agency Type', 'DistChannel', 'Product', 'Duration', 'Destination', 'NetSales', 'CommisionValue', 'Age','Claim']
df = pd.read_csv('train.csv',names=columns,skiprows=1)
df.head()


# In[186]:


df = df.drop(columns=['ID','Agency Type'])
df.head()


# In[187]:


X = df.drop(['Claim'],1)
y = df['Claim']
X_train, X_test, y_train, y_test= tts(X,y,random_state= 42, test_size = 0.25)
num_train = X_train.select_dtypes(include = np.number)
cat_train = X_train.select_dtypes(exclude = np.number)
num_test = X_test.select_dtypes(include = np.number)
cat_test = X_test.select_dtypes(exclude = np.number)
for x in list(cat_train):
    cat_train[x] = cat_train[x].str.lower()
for x in list(cat_test):
    cat_test[x] = cat_test[x].str.lower()

normalise = list(num_train)[0:-1]
minmax = list(num_train)[-1]

nor_scaler = StandardScaler().fit(num_train[normalise])
num1 = pd.DataFrame(nor_scaler.transform(num_train[normalise]), columns=normalise,index=num_train.index)
num3 = pd.DataFrame(nor_scaler.transform(num_test[normalise]), columns=normalise,index=num_test.index)

minmax_scalar = MinMaxScaler().fit(np.array(num_train[minmax]).reshape(-1,1))
num2 = pd.DataFrame(minmax_scalar.transform(np.array(num_train[minmax]).reshape(-1,1)).reshape(-1,),columns=[minmax],index=num_train.index)
num4 = pd.DataFrame(minmax_scalar.transform(np.array(num_test[minmax]).reshape(-1,1)).reshape(-1,),columns=[minmax],index=num_test.index)

num_train = pd.concat([num1,num2],axis=1)
num_test =pd.concat([num3,num4],axis=1)
X_train = pd.concat([num_train,cat_train],axis=1)
X_test = pd.concat([num_test,cat_test],axis=1)


# In[190]:


df = df[df['Duration']<=431]
df = df[df['Duration']>=0]
df = df[df['NetSales']<=507]
df = df[df['NetSales']>=0]
df = df.reset_index(drop=True)
X = df.drop(['Claim'],1)
y = df['Claim']
X_train, X_test, y_train, y_test= tts(X,y,random_state= 42, test_size = 0.25)
num_train = X_train.select_dtypes(include = np.number)
cat_train = X_train.select_dtypes(exclude = np.number)
num_test = X_test.select_dtypes(include = np.number)
cat_test = X_test.select_dtypes(exclude = np.number)
for x in list(cat_train):
    cat_train[x] = cat_train[x].str.lower()
for x in list(cat_test):
    cat_test[x] = cat_test[x].str.lower()

normalise = list(num_train)[0:-1]
minmax = list(num_train)[-1]

nor_scaler = StandardScaler().fit(num_train[normalise])
num1 = pd.DataFrame(nor_scaler.transform(num_train[normalise]), columns=normalise,index=num_train.index)
num3 = pd.DataFrame(nor_scaler.transform(num_test[normalise]), columns=normalise,index=num_test.index)

minmax_scalar = MinMaxScaler().fit(np.array(num_train[minmax]).reshape(-1,1))
num2 = pd.DataFrame(minmax_scalar.transform(np.array(num_train[minmax]).reshape(-1,1)).reshape(-1,),columns=[minmax],index=num_train.index)
num4 = pd.DataFrame(minmax_scalar.transform(np.array(num_test[minmax]).reshape(-1,1)).reshape(-1,),columns=[minmax],index=num_test.index)

num_train = pd.concat([num1,num2],axis=1)
num_test =pd.concat([num3,num4],axis=1)
X_train = pd.concat([num_train,cat_train],axis=1)
X_test = pd.concat([num_test,cat_test],axis=1)


# In[191]:


X_test


# In[ ]:




