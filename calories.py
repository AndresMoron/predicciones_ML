# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:18:36 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBRegressor 

df_exercise=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/exercise.csv")
print(df_exercise.info())
print(df_exercise.isnull().sum())
print(df_exercise.describe())

df_calories=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/calories.csv")
print(df_calories.info())
print(df_calories.isnull().sum())
print(df_calories.describe())

female_data={"Gender":{"female":1,"male":0}}
df_exercise.replace(female_data,inplace=True)

X=df_exercise.drop(columns="User_ID",axis=1)
y=df_calories.drop(columns="User_ID",axis=1)

sns.distplot(df_exercise["Height"])
sns.distplot(df_exercise["Weight"])
sns.distplot(df_exercise["Age"])
sns.distplot(df_exercise["Gender"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model=LinearRegression()
model.fit(X_train,y_train)

error_train_test=model.predict(X_train)
print("R2 error: ", metrics.r2_score(y_train, error_train_test))

error_test_test=model.predict(X_test)
print("R2 error: ", metrics.r2_score(y_test, error_test_test))

input_data=(0,68,190.0,94.0,29.0,105.0,40.8)

input_data_np = np.asarray(input_data)

input_data_np_reshape = input_data_np.reshape(1,-1)

pred=model.predict(input_data_np_reshape)

print("PREDICCION: ",pred)



#OTHER MODEL
model1=XGBRegressor()
model1.fit(X_train,y_train)

error_train_test=model1.predict(X_train)
print("R2 error: ", metrics.r2_score(y_train, error_train_test))

error_test_test=model1.predict(X_test)
print("R2 error: ", metrics.r2_score(y_test, error_test_test))

input_data=(0,68,190.0,94.0,29.0,105.0,40.8)

input_data_np = np.asarray(input_data)

input_data_np_reshape = input_data_np.reshape(1,-1)

pred=model.predict(input_data_np_reshape)

print("PREDICCION: ",pred)














