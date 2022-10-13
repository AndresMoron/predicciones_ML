# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:41:24 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/heart.csv")
df.info()
df.describe()

df["Sex"].value_counts()

print(df.select_dtypes(object))

for i in df.select_dtypes(object):
    print(df[i].value_counts())
    print("--------------------------")
    
sex_data={"Sex":{"M":0,"F":1}}
df.replace(sex_data,inplace=True)

chestpain_data = {"ChestPainType":{"ASY":0,"NAP":1,"ATA":2,"TA":3}}
df.replace(chestpain_data,inplace=True)

RestingECG_data={"RestingECG":{"Normal":0,"LVH":1,"ST":2}}
df.replace(RestingECG_data,inplace=True)

ExerciseAngina_data = {"ExerciseAngina":{"N":0,"Y":1}}
df.replace(ExerciseAngina_data,inplace=True)

ST_Slope_data = {"ST_Slope":{"Flat":0,"Up":1,"Down":2}}
df.replace(ST_Slope_data,inplace=True)

X=df.drop(columns=["HeartDisease"])
y=df["HeartDisease"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

x_accte= logistic.predict(X_test)
acctest = accuracy_score(x_accte,y_test)
print("Accuracy for Test Logistic: ",acctest)

x_acctr= logistic.predict(X_train)
acctrain = accuracy_score(x_acctr,y_train)
print("Accuracy for Train Logistic: ",acctrain)

input_data = (44,0,2,150,288,0,0,150,1,3,0)
input_data_np = np.asarray(input_data)
input_data_np_re = input_data_np.reshape(1,-1)

prediction = logistic.predict(input_data_np_re)
if prediction == 0:
    print("No tiene enfermedad del corazon")
else:
    print("Si tiene enfermedad del corazon")












