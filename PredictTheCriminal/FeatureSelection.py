#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:46:30 2018

@author: pinaki
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import OrderedDict

train_set=pd.read_csv("criminal_train.csv")
#test_set=pd.read_csv("criminal_test.csv")

X = train_set.iloc[:, 2:71].values
Y = train_set.iloc[:, 71].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train_all, X_test_all, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train_all = sc.fit_transform(X_train_all)
X_test_all = sc.transform(X_test_all)


performanceMonitors=[]
model = XGBClassifier()
for i in range(2,70):
    print(i)
    rfe = RFE(model, i)
    fit = rfe.fit(X_train_all, Y_train)
    ranking=fit.ranking_
    selectedParameters=np.flatnonzero((ranking == 1 ))
    X_train = X_train_all[:, selectedParameters]
    X_test  = X_test_all[:, selectedParameters]

    classifier = XGBClassifier()
    
    analyserObject=OrderedDict()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    analyserObject['ConfusionMatrix'] = confusion_matrix(Y_test, y_pred)
    analyserObject['Accuracy'] = accuracy_score(Y_test, y_pred)
    analyserObject["SelectedParameters"]=selectedParameters
    analyserObject["NumberOfparameters"]=fit.n_features_
    performanceMonitors.append(analyserObject)

performanceMonitors=sorted(performanceMonitors, key=lambda k: k['Accuracy'],reverse=True)
selectedParams=performanceMonitors[0]["SelectedParameters"]
len(selectedParams)
for i in selectedParams:
    print(i)