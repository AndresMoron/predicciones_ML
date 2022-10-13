# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:25:57 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


df= pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/real_estate_kaggle.csv")

df.info()
df.describe()

print(df.isnull().sum())

df=df[df["area"].notna()]
df=df[df["rooms"].notna()]
df=df[df["suites"].notna()]
df=df[df["bathrooms"].notna()]
df=df[df["parkings"].notna()]

plt.figure(figsize=(10,12))
sns.countplot(df["rooms"])
sns.countplot(df["suites"])
sns.countplot(df["bathrooms"])
sns.countplot(df["parkings"])


scaler=StandardScaler()

X=df.drop(columns=["price"],axis=1)
y=df["price"]
X_scaler = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaler,y,test_size=0.2,random_state=2)

linear=LinearRegression()
linear.fit(X_train,y_train)

x_trains= linear.predict(X_train)
score_train = r2_score(x_trains,y_train) 
print("r2 score para train es: ", score_train)

x_test= linear.predict(X_test)
score_test = r2_score(y_test,x_test) 
print("r2 score para train es: ", score_test)


tree=DecisionTreeRegressor()
tree.fit(X_train,y_train)

x_trains= tree.predict(X_train)
score_train = r2_score(x_trains,y_train) 
print("r2 score para train es: ", score_train)

x_test= tree.predict(X_test)
score_test = r2_score(y_test,x_test) 
print("r2 score para train es: ", score_test)


input_data = (136,3,1,3,1)
input_data_np= np.asarray(input_data)
input_data_np_re =input_data_np.reshape(1,-1)
input_scaler = scaler.fit_transform(input_data_np_re)


prediccion = tree.predict(input_scaler)

print(prediccion)

 










