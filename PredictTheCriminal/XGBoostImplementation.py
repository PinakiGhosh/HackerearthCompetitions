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

selectedParams=[0,3,8,13,16,26,42,54,55,59,66]
X_train = train_set.iloc[:, selectedParams].values
Y_train = train_set.iloc[:, 71].values

X_test = test_set.iloc[:, selectedParams].values

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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