# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:34:32 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn import metrics

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/car data.csv")
print(df.info())
print(df.isnull().sum())

sns.countplot(df["Fuel_Type"])
sns.countplot(df["Seller_Type"])
sns.countplot(df["Transmission"])


l=list(df["Car_Name"].unique())
plt.figure(figsize=(22,9))
q=sns.countplot(df["Car_Name"])
q.set_xticklabels(labels=l,rotation=90)
sns.set(rc={"figure.figsize":(12,9)})

sns.distplot(df["Present_Price"])
sns.distplot(df["Selling_Price"])
sns.distplot(df["Kms_Driven"])

print(df["Fuel_Type"].unique())
print(df["Transmission"].unique())

fuel_data={"Fuel_Type":{
    "Petrol":0,
    "Diesel":1,
    "CNG":2}}
df.replace(fuel_data,inplace=True)

seller_data={"Seller_Type":{
    "Dealer":0,
    "Individual":1
    }}
df.replace(seller_data,inplace=True)

Transmission_data={"Transmission":{
    "Manual":0,
    "Automatic":1
    }}
df.replace(Transmission_data,inplace=True)

X=df.drop(columns=["Car_Name","Selling_Price"])
y=df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2)

### LINEAR MODEL ###
lineal=LinearRegression()
lineal.fit(X_train,y_train)

test_lineal_accuracy = lineal.predict(X_test)
print("Accuracy r2:", metrics.r2_score(y_test, test_lineal_accuracy))

train_lineal_accuracy = lineal.predict(X_train)
print("Accuracy r2:", metrics.r2_score(y_train, train_lineal_accuracy))

### LASSO MODEL ###
lasso=Lasso()
lasso.fit(X_train,y_train)

test_lasso_accuracy = lasso.predict(X_test)
print("Accuracy r2:", metrics.r2_score(y_test, test_lasso_accuracy))

train_lasso_accuracy = lasso.predict(X_train)
print("Accuracy r2:", metrics.r2_score(y_train, train_lasso_accuracy))

"""
plt.plot(y_test,color="red",label="actual price")
plt.plot(test_lasso_accuracy,color="blue",label="predict price")
plt.xlabel("num valores")
plt.ylabel("Precio")
"""
eje_x=list(range(X_test.shape[0]))
eje_y=y_test
eje_y2=test_lineal_accuracy

fig,ax = plt.subplots()
ax.plot(eje_x,eje_y,"-",eje_y2,"o")
fig.set_size_inches(15,9)


input_data = (2003,2.25,62000,0,0,0,0)
input_data_np=np.asarray(input_data)
input_data_np_resh= input_data_np.reshape(1,-1)

print("Prediccion: ", lasso.predict(input_data_np_resh))







