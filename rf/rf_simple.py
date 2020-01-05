# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("../train.csv")


df =  df.drop(["ID"],1)
le = LabelEncoder()
df["Agency"] = le.fit_transform(df["Agency"])
df["Agency Type"] = le.fit_transform(df["Agency Type"])
df["Distribution Channel"] = le.fit_transform(df["Distribution Channel"])
df["Product Name"] = le.fit_transform(df["Product Name"])
df["Destination"] = le.fit_transform(df["Destination"])
    

X = df.drop(["Claim"],1)
y = df["Claim"]
X_train, X_test, y_train, y_test = tts(X,y,random_state = 42, test_size = 0.25, stratify = y)


#model
rfc = RandomForestClassifier(random_state=42, criterion = "entropy", max_depth=6, min_samples_split=0.12, 
                            oob_score = True, n_estimators = 40, class_weight = "balanced")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
rfc.score(X_test,y_test)
print (classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)