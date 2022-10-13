# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 23:14:39 2022

@author: andre
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/cancer.csv")
print(df.info())
print(df.isnull().sum())

diagnostic_data={"diagnosis":{
    "B":0,
    "M":1}}

df.replace(diagnostic_data,inplace=True)

sns.distplot(df["diagnosis"])
sns.distplot(df["compactness_mean"])

group=df.groupby("diagnosis").mean()

X=df.drop(columns=["id","diagnosis","Unnamed: 32"])
y=df["diagnosis"]

X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2,random_state=2)

### SVM MODELO ###
vector= svm.SVC(kernel=("linear"))
vector.fit(X_train,y_train)

test_accuracy=vector.predict(X_test)
print("Accuracy for TEST: ", accuracy_score(y_test,test_accuracy))

train_accuracy=vector.predict(X_train)
print("Accuracy for train: ", accuracy_score(y_train,train_accuracy))



### Logistic MODELO ###
logistic = LogisticRegression()
logistic.fit(X_train,y_train)

test_accuracy=logistic.predict(X_test)
print("Accuracy for TEST: ", accuracy_score(y_test,test_accuracy))

train_accuracy=logistic.predict(X_train)
print("Accuracy for train: ", accuracy_score(y_train,train_accuracy))

### DECISION TREE MODELO ###
arbol =DecisionTreeClassifier()
arbol.fit(X_train,y_train)

test_accuracy=arbol.predict(X_test)
print("Accuracy for TEST: ", accuracy_score(y_test,test_accuracy))

train_accuracy=arbol.predict(X_train)
print("Accuracy for train: ", accuracy_score(y_train,train_accuracy))




input_data=(7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039)

input_data_np = np.asarray(input_data)

input_data_np_resh= input_data_np.reshape(1,-1)

predict=arbol.predict(input_data_np_resh)

if predict == 1:
    print("Tiene tumor maligno")
else:
    print("Tiene tumor benigno")