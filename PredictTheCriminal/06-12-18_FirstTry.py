#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:13:17 2018

@author: pinaki
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv("criminal_train.csv")

X = dataset.iloc[:, 2:71].values
y = dataset.iloc[:, 71].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from collections import OrderedDict

def analyseModel(classifier,X_Test,Y_Test,X_Train,Y_Train,runCrossVal=False,cv=10):
    analyserObject=OrderedDict()
    analyserObject['ClassifierType']=str(type(classifier))
    classifier.fit(X_Train, Y_Train)
    y_pred = classifier.predict(X_Test)
    analyserObject['ConfusionMatrix'] = confusion_matrix(Y_Test, y_pred)
    analyserObject['Accuracy'] = accuracy_score(Y_Test, y_pred)
    if runCrossVal:
        accuracies = cross_val_score(estimator = classifier, X = X_Train, y = Y_Train, cv = cv)
        analyserObject['AccuracyList'] = accuracies
        analyserObject['MeanAccuracy'] = accuracies.mean()
        analyserObject['AccuracySD'] = accuracies.std()
    return analyserObject

#List of performance monitors
performanceMonitors=[]

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting LogisticRegression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting KNeighborsClassifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting Kernel SVN to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting Naiyesve Ba to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting DecisionTree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

performanceMonitors=sorted(performanceMonitors, key=lambda k: k['MeanAccuracy'],reverse=True)


for i in performanceMonitors:
    cm=i["ConfusionMatrix"]
    print(cm[1][1],cm[0][1],cm[1][0],i["ClassifierType"])