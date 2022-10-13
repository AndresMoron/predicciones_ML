# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:40:59 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Copy_of_sonar_data.csv")
print(df.info())
print(df.isnull().sum())

print(df.shape)

columnas=[]
for i in range(1,63):
    columnas.append(i)
    print(columnas-1)
    
df.columns=columnas
    

print(df.describe())

data_r = {61:{"R":0,"M":1}}
df.replace(data_r,inplace=True)

group=df.groupby(61).mean()

X=df.drop(columns=61)
y=df[61]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
logistic=LogisticRegression()
logistic.fit(X_train,y_train)

test_accuracy = logistic.predict(X_test)
print("Accuracy for Test: ",accuracy_score(y_test, test_accuracy))

train_accuracy = logistic.predict(X_train)
print("Accuracy for Test: ",accuracy_score(y_train, train_accuracy))

input_data=(0.0414,0.0436,0.0447,0.0844,0.0419,0.1215,0.2002,0.1516,0.0818,0.1975,0.2309,0.3025,0.3938,0.5050,0.5872,0.6610,0.7417,0.8006,0.8456,0.7939,0.8804,0.8384,0.7852,0.8479,0.7434,0.6433,0.5514,0.3519,0.3168,0.3346,0.2056,0.1032,0.3168,0.4040,0.4282,0.4538,0.3704,0.3741,0.3839,0.3494,0.4380,0.4265,0.2854,0.2808,0.2395,0.0369,0.0805,0.0541,0.0177,0.0065,0.0222,0.0045,0.0136,0.0113,0.0053,0.0165,0.0141,0.0077,0.0246,0.0198)

input_data_np=np.asarray(input_data)

input_data_np_res=input_data_np.reshape(1,-1)

predict= logistic.predict(input_data_np_res)

if predict==0:
    print("Es una piedra mas bien")
else:
    print("Rezale al de arriba")





