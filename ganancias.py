# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:08:57 2022

@author: andre
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

columnas=["age","workclass","fnlwgt","education","education-num",
          "marital-status","occupation","relationship","race","sex",
          "capital-gain","capital-loss","hours-per-week","native-country","target"]

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/adult.csv")

print(df.info())

print(df.isnull().sum())

df.columns=columnas
print(df["target"].unique())
print(df["race"].unique())
print(df["sex"].unique())
print(df["relationship"].unique())
print(df["education"].unique())
print(df["marital-status"].unique())

target_data={"target":{
    " <=50K":0,
    " >50K":1
    }}


df.replace(target_data,inplace=True)

race_data={"race":{
    " White":0,
    " Black":1,
    " Asian-Pac-Islander":2,
    " Amer-Indian-Eskimo":3,
    " Other":4
    }}


df.replace(race_data,inplace=True)

sex_data={"sex":{
    " Male":0,
    " Female":1
    }}


df.replace(sex_data,inplace=True)

relationship_data={"relationship":{
    " Husband":0,
    " Not-in-family":1,
    " Wife":2,
    " Own-child":3,
    " Unmarried":4,
    " Other-relative":5
    }}

df.replace(relationship_data,inplace=True)


education_data={"education":{
    " Bachelors":0,
    " HS-grad":1,
    " 11th":2,
    " Masters":3,
    " 9th":4,
    " Some-college":5,
    " Assoc-acdm":6,
    " Assoc-voc":7,
    " 7th-8th":8,
    " Doctorate":9,
    " Prof-school":10,
    " 5th-6th":11,
    " 10th":12,
    " 1st-4th":13,
    " Preschool":14,
    " 12th":15
    }}

df.replace(education_data,inplace=True)

marital_status_data={"marital-status":{
    " Married-civ-spouse":0,
    " Divorced":1,
    " Married-spouse-absent":2,
    " Never-married":3,
    " Separated":4,
    " Married-AF-spouse":5,
    " Widowed":6
    }}

df.replace(marital_status_data,inplace=True)

print(df["workclass"].value_counts())
print("=============================")
print(df["occupation"].value_counts())
print("=============================")
print(df["native-country"].value_counts())

workclass_nan={"workclass":{
    "Nan":" Private"}}
df.replace(workclass_nan,inplace=True)


workclass_place=df.pivot_table(values="workclass",columns="fnlwgt",aggfunc=(lambda x:x.mode()[0]))
print(workclass_place)

workclass_education=df.pivot_table(values="workclass",columns="education-num",aggfunc=(lambda x:x.mode()[0]))
print(workclass_education)

occupation_place=df.pivot_table(values="occupation",columns="fnlwgt",aggfunc=(lambda x:x.mode()[0]))
print(occupation_place)

occupation_education=df.pivot_table(values="occupation",columns="education-num",aggfunc=(lambda x:x.mode()[0]))
print(occupation_education)

occupation_nan={"occupation":{
    " ?":" Other-service"}}
df.replace(occupation_nan,inplace=True)


country_place=df.pivot_table(values="native-country",columns="fnlwgt",aggfunc=(lambda x:x.mode()[0]))
print(country_place)

country_education=df.pivot_table(values="native-country",columns="education-num",aggfunc=(lambda x:x.mode()[0]))
print(country_education)

country_nan={"native-country":{
    " ?":" United-States"}}
df.replace(country_nan,inplace=True)


l=list(df["native-country"].unique())
q=sns.countplot(df["native-country"])
q.set_xticklabels(labels=l,rotation=90)

l=list(df["workclass"].unique())
x1=sns.countplot(df["workclass"])
x1.set_xticklabels(labels=l,rotation=90)

sns.countplot(df["sex"])

sex_target=df.groupby("target").mean()

print(df["sex"].value_counts())


encoder=LabelEncoder()
df["occupation"]=encoder.fit_transform(df["occupation"])
df["native-country"]=encoder.fit_transform(df["native-country"])
df["workclass"]=encoder.fit_transform(df["workclass"])

X=df.drop(columns="target")
y=df["target"]


scaler=StandardScaler()
X=scaler.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=2)
print(X_train.shape)
print(X_test.shape)

### MODEL XGBRegressor ###
model=XGBRegressor()
model.fit(X_train,y_train)

test_pred= model.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred))

train_pred= model.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred))



### MODEL LogisticRegression ###
logistic=LogisticRegression()
logistic.fit(X_train,y_train)

test_pred_Logi= logistic.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred_Logi))

train_pred_Logi= logistic.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred_Logi))

### MODEL LinearRegression ###
linear=LinearRegression()
linear.fit(X_train,y_train)

test_pred_Line= linear.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred_Line))

train_pred_Line= linear.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred_Line))

### MODEL XGBClassification ###
xgboost=XGBClassifier()
xgboost.fit(X_train,y_train)

test_pred_Boost= xgboost.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred_Boost))

train_pred_Boost= xgboost.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred_Boost))

### MODEL SVM ###
vector=svm.SVC(kernel=("linear"))
vector.fit(X_train,y_train)

test_pred_Vec= vector.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred_Vec))

train_pred_Vec= vector.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred_Vec))

### MODEL Decision Tree Classifier ###
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

test_pred_tree= tree.predict(X_test)
print("Accuracy for Test: ",metrics.r2_score(y_test, test_pred_tree))

train_pred_tree= tree.predict(X_train)
print("Accuracy for Train: ",metrics.r2_score(y_train, train_pred_tree))




### PREDICTION ###
input_data=(52,5,287927,9,9,0,2,2,0,1,15024,0,40,38)

input_data_np = np.asarray(input_data)

input_data_np= input_data_np.reshape(1,-1)

scaler_data= scaler.transform(input_data_np)

predict=tree.predict(scaler_data)

print("Prediccion: ", predict)




















