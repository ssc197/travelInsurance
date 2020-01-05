#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:27:25 2020

@author: zaonx
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle

columns = ['ID','Agency','Agency Type', 'DistChannel', 'Product', 'Duration', 'Destination', 'NetSales', 'CommisionValue', 'Age','Claim']
df = pd.read_csv('data/train.csv',names=columns,skiprows=1)


# =============================================================================
# df =  df.drop(["ID"],1)
# =============================================================================


le = LabelEncoder()
df["Agency"] = le.fit_transform(df["Agency"])
df["Agency Type"] = le.fit_transform(df["Agency Type"])
df["DistChannel"] = le.fit_transform(df["DistChannel"])
df["Product"] = le.fit_transform(df["Product"])
df["Destination"] = le.fit_transform(df["Destination"])
    
# removing duration outliear
#print((df["Duration"]<=0).value_counts())
# =============================================================================
# df.drop(df[df["Duration"]<=0].index, inplace = True)
# df.drop(df[df["Duration"]>=4000].index, inplace = True)
# =============================================================================
# end

X = df.drop(["Claim"],1)
y = df["Claim"]
X_train, X_test, y_train, y_test = tts(X,y,random_state = 42, test_size = 0.25, stratify = y)


#model
from utils import setModel

# =============================================================================
# model = setModel("dtc")
# model.fit(X,y)
# pickle.dump(model, open('dtc.pkl','wb'))
# 
# model = setModel("log_cls")
# model.fit(X,y)
# pickle.dump(model, open('log_cls.pkl','wb'))
# 
# model = setModel("ran_for")
# model.fit(X,y)
# pickle.dump(model, open('ran_for.pkl','wb'))
# =============================================================================

model = setModel("knn_cls")
model.fit(X,y)
pickle.dump(model, open('knn_cls.pkl','wb'))


print (model.best_estimator_)
y_pred = model.predict(X_test)
print (classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
# =============================================================================
# rfc = RandomForestClassifier(random_state=42, criterion = "entropy", max_depth=6, min_samples_split=0.12, 
#                             oob_score = True, n_estimators = 40, class_weight = "balanced")
# rfc.fit(X_train,y_train)
# y_pred = rfc.predict(X_test)
# rfc.score(X_test,y_test)
# print (classification_report(y_test,y_pred))
# accuracy_score(y_test,y_pred)
# =============================================================================
