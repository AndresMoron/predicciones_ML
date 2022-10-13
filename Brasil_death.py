# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:07:43 2021

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/death_cause_brazil.csv")
print(df.info())

print(df["age"].unique())
print(df["cause"].unique())
print(df["color"].unique())
print(df["state"].unique())

nuevo_age={"age":{
    "< 9":0,
    "10 - 19":1,
    "20 - 29":2,
    "30 - 39":3,
    "40 - 49":4,
    "50 - 59":5,
    "60 - 69":6,
    "70 - 79":7,
    "80 - 89":8,
    "90 - 99":9,
    "> 100":10,
    "N/I":11,
    }}

df.replace(nuevo_age,inplace=True)

nueva_muerte={"cause":{
    'Septicemia' :0,
    'Hearth attack' :1,
    'Others' :2,
    'Cardiogenic shock' :3,
    'Pneumonia':4,
 'Stroke' :5,
 'Undetermined' :6,
 'Respiratory failure' :7,
 'Cardiopathy':8,
 'Sudden death' :9,
 'Sars' :10,
 'Covid' :11,
 'Covid (stroke)' :12,
 'Covid (hearth attack)':13,
 'Unknown':14,
    
    }}

df.replace(nueva_muerte,inplace=True)

nuevo_estado={"state":{
    'AC':0, 'AL':1, 'AM':2, 'AP':3, 'BA':4, 'CE':5, 'DF':6, 
    'ES':7, 'GO':8, 'MA':9, 'MG':10, 'MS' :11,
    'MT':12, 'PA':13, 'PB':14, 'PE':15, 'PI':16, 'PR':17, 
    'RJ':18, 'RN':19, 'RO':20, 'RR':21, 'RS':22, 'SC' :23,
    'SE' :24,'SP' :25,'TO':26,
    
    }}

df.replace(nuevo_estado,inplace=True)


df=df.drop(columns=["color"])
df=df.drop(columns=["date"])
df=df.drop(columns=["total"])
df=df.drop(columns=["gender"])

df["state"]=pd.to_numeric(df["state"] )

print(df.info())

######################################
X_datos=df[["age","cause"]]

Y_datos=df[["state"]]

X_test,X_train,y_test,y_train=train_test_split(X_datos,Y_datos,test_size=0.2)

modelo=LinearRegression()
modelo.fit(X_train,y_train)

prediccion=modelo.predict(X_test)

error=np.sqrt(mean_squared_error(y_test, prediccion))
print(error*100)










"""
brasil=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/death_cause_brazil.csv")
print(brasil["age"].unique())
print(brasil["cause"].unique())
print(brasil["color"].unique())
print(brasil["state"].unique())

nuevo_age={"age":{
    "< 9":0,
    "10 - 19":1,
    "20 - 29":2,
    "30 - 39":3,
    "40 - 49":4,
    "50 - 59":5,
    "60 - 69":6,
    "70 - 79":7,
    "80 - 89":8,
    "90 - 99":9,
    "> 100":10,
    "N/I":11,
    }}

brasil.replace(nuevo_age,inplace=True)

nueva_muerte={"cause":{
    'Septicemia' :0,
    'Hearth attack' :1,
    'Others' :2,
    'Cardiogenic shock' :3,
    'Pneumonia':4,
 'Stroke' :5,
 'Undetermined' :6,
 'Respiratory failure' :7,
 'Cardiopathy':8,
 'Sudden death' :9,
 'Sars' :10,
 'Covid' :11,
 'Covid (stroke)' :12,
 'Covid (hearth attack)':13,
 'Unknown':14,
    
    }}

brasil.replace(nueva_muerte,inplace=True)

nuevo_estado={"state":{
    'AC':0, 'AL':1, 'AM':2, 'AP':3, 'BA':4, 'CE':5, 'DF':6, 
    'ES':7, 'GO':8, 'MA':9, 'MG':10, 'MS' :11,
    'MT':12, 'PA':13, 'PB':14, 'PE':15, 'PI':16, 'PR':17, 
    'RJ':18, 'RN':19, 'RO':20, 'RR':21, 'RS':22, 'SC' :23,
    'SE' :24,'SP' :25,'TO':26,
    
    }}

brasil.replace(nuevo_estado,inplace=True)


brasil=brasil.drop(columns=["color"])
brasil=brasil.drop(columns=["date"])
brasil=brasil.drop(columns=["total"])
brasil=brasil.drop(columns=["gender"])

brasil["state"]=pd.to_numeric(brasil["state"] )

print(brasil.info())
#########################################################

datos_entrenamiento=brasil.sample(frac=0.8,random_state=0)
datos_test=brasil.drop(datos_entrenamiento.index)

etiqueta_entrenamiento=datos_entrenamiento.pop("state")
etiqueta_test=datos_test.pop("state")

modelo=LinearRegression()
modelo.fit(datos_entrenamiento,etiqueta_entrenamiento)

prediccion= modelo.predict(datos_test)

error=np.sqrt(mean_squared_error(etiqueta_test,prediccion))
print("Error:", error*100)

nueva_prediccion=pd.DataFrame(np.array([[5,8]]), columns=["age","cause"])

print(modelo.predict(nueva_prediccion))


"""




































