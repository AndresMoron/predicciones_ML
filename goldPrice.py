# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 00:18:16 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/gld_price_data.csv")
print(df.info())
print(df.isnull().sum())
print(df.describe())


#Correlation
#Positive correlation
#Negative correlation

correlation= df.corr()

#Constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cmap="Blues",fmt=".1f",square="True",annot=True,annot_kws={"size":8})

#Correlation values of GOLD
print(correlation["GLD"])

#Checking the distribution of the GOLD
sns.distplot(df["GLD"])

#Splitting the Features
X=df.drop(["Date","GLD"],axis=1)
y=df["GLD"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#Model Training
regressor=RandomForestRegressor(n_estimators=100)

regressor.fit(X_train,y_train)

#Model evaluation

#Prediction on Test Data
test_data_prediction=regressor.predict(X_test)
error_test_data=metrics.r2_score(y_test, test_data_prediction)
print(error_test_data)


train_data_prediction=regressor.predict(X_train)
error_train_data=metrics.r2_score(y_train, train_data_prediction)
print(error_train_data)

#Compare the actual values and predicted values in a PLOT
y_test=list(y_test)

plt.plot(y_test,color="red",label="Actual price")
plt.plot(test_data_prediction,color="blue",label="Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.xlabel("Number of values")
plt.ylabel("Gold price")
plt.legend()






















