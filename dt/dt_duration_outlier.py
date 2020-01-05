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

# removing duration outliear
#print((df["Duration"]<=0).value_counts())
df.drop(df[df["Duration"]<=0].index, inplace = True)
df.drop(df[df["Duration"]>=4000].index, inplace = True)
# end
X = df.drop(["Claim"],1)
y = df["Claim"]
X_train, X_test, y_train, y_test = tts(X,y,random_state = 42, test_size = 0.25, stratify = y)


#model
dtc = DecisionTreeClassifier(random_state=42)
# dtc = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=10, min_samples_split=0.06,class_weight="balanced")
dtc.fit(X_train,y_train)
dtc.score(X_test,y_test)
y_pred = dtc.predict(X_test)
print (classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)