#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 22:56:23 2018

@author: pinaki
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

train_set=pd.read_csv("criminal_train.csv")
test_set=pd.read_csv("criminal_test.csv")

#selectedParams=[0,3,8,13,16,26,42,54,55,59,66]
#selectedParams=[13,26,54,55]
selectedParams=list(range(2,70))
X_train = train_set.iloc[:, selectedParams].values
Y_train = train_set.iloc[:, 71].values

X_test = test_set.iloc[:, selectedParams].values

# Feature Scaling is not required for XGBoost
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# TODO : Need to do parameter tuning -> http://xgboost.readthedocs.io/en/latest/python/python_api.html
# Applying Grid Search to find the best model and the best parameters
'''
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              # My addition{'C': [1, 10, 100, 1000], 'kernel': ['sigmoid','rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
             ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''

from xgboost import XGBClassifier
classifier = XGBClassifier()

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

f=open("Output_XGBoost.csv","w")
f.write("PERID,Criminal\n")
count=0
perIds=list(test_set["PERID"])
for i in range(0,len(perIds)):
    f.write(str(perIds[i]))
    f.write(",")
    f.write(str(Y_pred[count]))
    f.write("\n")
    count=count+1
f.close()