# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:39:15 2020

@author: Sharu
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")
#df = pd.read_csv("train.csv")

Models = {
            'ran_for':{
                        'model':RandomForestClassifier(random_state=42, criterion = "entropy", max_depth=6, min_samples_split=0.12, 
                            oob_score = True, n_estimators = 40, class_weight = "balanced"),
                        'params':{
                                'n_estimators'      : [320,330,340],
                                'max_depth'         : [8, 9, 10, 11, 12],
                                'random_state'      : [0]
                            }
                    },
             'dtc':{
                        'model':DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=0.06,class_weight="balanced"),
                        'params':[{'criterion': ['entropy', 'gini'], 'max_depth': np.arange(1, 10, 1)},
                                   {'min_samples_leaf': np.arange(1, 10, 1)}]},
             'log_cls':{
                        'model':LogisticRegression(random_state=42),
                        'params':{"C":np.arange(0.1,10,0.1),"penalty":["l1", "l2"]}
                    },
             'knn_cls':{
                        'model':KNeighborsClassifier(),
                        'params': {"n_neighbors":np.arange(1,228,1), "metric":["euclidean", "minkowski", "jaccard", "cosine", "manhattan"]}
                    }
        }

def extract_best_features(model,_X, n):
    print(model)
    features = list(_X)
    fs = pd.DataFrame()
    try:
        ranking = pd.Series(model.coef_[0])
    except:
        print(model.feature_importances_)
        ranking = pd.Series(model.feature_importances_)
    fs["features"] = features
    fs["ranking"] = ranking
    fs["ranking"] = np.abs(fs["ranking"])
    fs = fs.sort_values(["ranking"], ascending=False)
    fs = fs[:n]
    return fs

def setModel(_model):
    _myModel = Models[_model]['model']
    _myModel_cv = GridSearchCV(param_grid= Models[_model]['params'], cv = 5, estimator=_myModel )
    return _myModel_cv


    
    
