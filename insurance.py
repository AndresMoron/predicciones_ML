# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:54:03 2022

@author: andre
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

### MODEL SVM
df = pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/insurance.csv")
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df["region"].unique())



mean_sex= df.groupby("sex").mean()

sex_value={"sex":{
    "male":0,
    "female":1}}

df.replace(sex_value,inplace=True)

somke_value={"smoker":
             {"yes":1,
              "no":0}}

df.replace(somke_value,inplace=True)

region_value={"region":{
    "southwest":0,
    "southeast":1,
    "northwest":2,
    "northeast":3}}

df.replace(region_value,inplace=True)

#Escalando los datos
X=df.drop(columns="charges",axis=1)
y=df["charges"].astype("int")

scaler=StandardScaler()
standirized_data=scaler.fit_transform(X)
X=standirized_data




#Training

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)


model=svm.SVC(kernel="linear")
model.fit(X_train,y_train)

X_train_prediction=model.predict(X_train)
X_train_accuracy = accuracy_score(y_train,X_train_prediction)
print(" Training Accuracy score: ",X_train_accuracy )

metrics_accuracy = metrics.r2_score(y_train, X_train_prediction)
print("Accuracy score of training data METRICS TRAIN: ", metrics_accuracy)


X_test_prediction=model.predict(X_test)
X_test_accuracy = accuracy_score(y_test, X_test_prediction)
print("Test Accuracy score: ",X_test_accuracy )

metrics_accuracy = metrics.r2_score(y_test, X_test_prediction)
print("Accuracy score of training data METRICS TEST: ", metrics_accuracy)


input_data = (30,0,27.645,1,0,3)

#Changing the input_data to numpy array
input_data_as_np = np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_np.reshape(1,-1)

#Standarize the input data
std_data = scaler.transform(input_data_reshaped)

print(std_data)

x=model.predict(std_data)
print("==================================================")
print("Insurence prediction: ",x)




























"""
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/insurance.csv")
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df["region"].unique())
print(df.describe())

sex_num={"sex":{
    "female":0,
    "male":1}}

df.replace(sex_num,inplace=True)

smoker_num={"smoker":{
    "yes":0,
    "no":1}}

df.replace(smoker_num,inplace=True)

region_num={"region":{
    "southwest":0,
    "southeast":1,
    "northwest":2,
    "northeast":3}}

df.replace(region_num,inplace=True)
sns.set()
sns.distplot(df["region"])
sns.distplot(df["children"])
sns.distplot(df["smoker"])
sns.distplot(df["charges"])
sns.set()
sns.distplot(df["age"])

money= df.groupby("age").mean()



X= df.drop(columns="charges",axis=1)
y=df["charges"]

scaler=StandardScaler()
standarized_data = scaler.fit_transform(X)
X=standarized_data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

regression=LinearRegression()
regression.fit(X_train,y_train)

predict = regression.predict(X_train)

r2 = metrics.r2_score(predict, y_train)
print(r2)


input_data = (18,1,33.77,1,1,1)

#Changing the input_data to numpy array
input_data_as_np = np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_np.reshape(1,-1)

#Standarize the input data
std_data = scaler.transform(input_data_reshaped)

print(std_data)

x=regression.predict(std_data)

print("Insurence: ",x)











df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/insurance.csv")
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df["region"].unique())


sex_num={"sex":{
    "female":0,
    "male":1}}

df.replace(sex_num,inplace=True)

smoker_num={"smoker":{
    "yes":0,
    "no":1}}

df.replace(smoker_num,inplace=True)

region_num={"region":{
    "southwest":0,
    "southeast":1,
    "northwest":2,
    "northeast":3}}

df.replace(region_num,inplace=True)

sns.distplot(df["region"])
sns.distplot(df["children"])
sns.distplot(df["smoker"])
sns.distplot(df["charges"])

money= df.groupby("age").mean()


X= df.drop(columns="charges")
y=df["charges"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

regression=LinearRegression()
regression.fit(X_train,y_train)

predict = regression.predict(X_train)
predict_test = regression.predict(X_test)

r2 = metrics.r2_score(predict, y_train)
print(r2)

r2 = metrics.r2_score(predict_test, y_test)
print(r2)


nuevo = pd.DataFrame(np.array([[31,1,25.74,0,1,0]]),columns=("age","sex","bmi","children",
                                                              "smoker","region"))

x=regression.predict(nuevo)
print(x)


### Crossplot Actual vs Prediction

plt.scatter(y_train,predict)
plt.xlabel("Actual")
plt.ylabel("Prediccion")
plt.grid(True)

print(df["charges"].describe())

"""