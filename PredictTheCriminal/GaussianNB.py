#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:14:10 2018

@author: pinaki
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

train_set=pd.read_csv("criminal_train.csv")
test_set=pd.read_csv("criminal_test.csv")

X_train = train_set.iloc[:, 2:71].values
Y_train = train_set.iloc[:, 71].values

X_test = test_set.iloc[:, 2:71].values

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

from xgboost import XGBClassifier
classifier = XGBClassifier()

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

f=open("Output_Gaussian_NB.csv","w")
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


