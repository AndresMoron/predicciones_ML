# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:01:35 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/cancer.csv")

print(df.shape)

print(df.describe())

print(df.info())

diagnosis_data={"diagnosis":{"M":0,"B":1}}
df.replace(diagnosis_data,inplace=True)

sns.displot(df["diagnosis"])

X=df.drop(columns=["id","diagnosis","Unnamed: 32"])
y=df["diagnosis"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

### LOGISTIC REGRESSION
logistic=LogisticRegression()
logistic.fit(X_train,y_train)

train_test_accuracy = logistic.predict(X_train)
train_accuracy = accuracy_score(train_test_accuracy, y_train)
print("TRAIN ACCURACY: ", train_accuracy)

test_test_accuracy = logistic.predict(X_test)
test_accuracy = accuracy_score(test_test_accuracy,y_test)
print("TEST ACCURACY: ", test_accuracy)

### SVM

vector=svm.SVC(kernel=("linear"))
vector.fit(X_train,y_train)

test_svm=vector.predict(X_test)
test_svm_accuracy= accuracy_score(test_svm, y_test)
print(" SVM PREDICT TEST: ",test_svm_accuracy)

train_svm = vector.predict(X_train)
train_svm_accuracy = accuracy_score(train_svm, y_train)
print(" SVM TRAIN ACCURACY: ", train_svm_accuracy)

### Decision Tree
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

test_tree= tree.predict(X_test)
test_tree_acc = accuracy_score(test_tree,y_test)
print("TREE TEST ACCURACY: ", test_tree_acc)

train_tree = tree.predict(X_train)
train_tree_acc = accuracy_score(train_tree,y_train)
print("TREE TRAIN ACCURACY: ", train_tree_acc)

input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)

input_data_np= np.asarray(input_data)

input_data_np_res= input_data_np.reshape(1,-1)

prediccion=tree.predict(input_data_np_res)

print(prediccion)