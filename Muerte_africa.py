# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:32:25 2021

@author: andre

"""
#SEGUNDA FORMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

africa=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/phpgNaXZe.csv")
columns=["sbp","Tabaco","Ldl","Adiposity","Familia","Tipo",
         "Obesidad","Alcohol","Edad","EC",]


africa.columns=columns

print(africa.isnull().sum())

#Corregir valores de la columna familia y chd ya que son 1 y 2
from sklearn.preprocessing import LabelEncoder #Codifica resultados de una etiqueta

encoder=LabelEncoder()
africa["Familia"]=encoder.fit_transform(africa["Familia"])
africa["EC"]=encoder.fit_transform(africa["EC"])
print(africa.head())

#Ahora hay datos que son muy elevados, vamos a escalarlos a la medida normal de todos los
#datos para un mejor resultado
from sklearn.preprocessing import MinMaxScaler #Establecemos un rango min y max definido

#Definimos los rangos
scale=MinMaxScaler(feature_range=(0,100))
africa["sbp"]=scale.fit_transform(africa["sbp"].values.reshape(-1,1))
print(africa.head())

africa.plot(x="Edad",y="Obesidad",kind="scatter",figsize=(10,5))

africa.plot(x="Edad",y="Tabaco",kind="scatter",figsize=(10,6))

africa.plot(x="Edad",y="Alcohol",kind="scatter",figsize=(10,5))

#Construir el modelo ML
from sklearn.model_selection import train_test_split  #Para separar datos de entrenamiento y pruebas
from sklearn import svm  #Algoritmo a usar
#Evaluacion del modelo
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import mean_squared_error

#Definir las variables dependientes e independientes
y=africa["EC"]
x=africa.drop("EC",axis=1)

#Separamos datos de entrenamiento y prueba
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)

#Definimos algoritmo a utilizar
algoritmo = svm.SVC(kernel="linear")

algoritmo.fit(X_train, y_train)

prediccion=algoritmo.predict(X_test)

print(confusion_matrix(y_test, prediccion))

error=mean_squared_error(y_test, prediccion)
print(error*100)

print(accuracy_score(y_test, prediccion))
print(precision_score(y_test, prediccion))
"""
#PRIMERA FORMA
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

africa=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/phpgNaXZe.csv")
print(africa.keys())
print(africa.info())
#Enfocarse en V1,V2,V3,V7,V8,V9

africa=africa.drop(columns=["V4"])
africa=africa.drop(columns=["V5"])
africa=africa.drop(columns=["V6"])

plt.scatter(x=africa["V7"], y=africa["Class"])

X_train= africa.sample(frac=0.8, random_state=0)
X_test= africa.drop(X_train.index)

y_train= X_train.pop("Class")
y_test= X_test.pop("Class")

modelo=LinearRegression()
modelo.fit(X_train,y_train)

prediccion=modelo.predict(X_train)
print(prediccion)

nueva_persona=pd.DataFrame(np.array([[150,14,5,33,97,52]]),columns=["V1","V2","V3","V7","V8","V9",])
print(nueva_persona)

print(modelo.predict(nueva_persona))
"""