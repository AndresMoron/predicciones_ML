# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:38:16 2021

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error

#Importamos el csv
helado=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/datos_ventas.csv")
print(helado.keys())

print(helado.hist())

X_train= helado.sample(frac=0.8, random_state=0)
X_test= helado.drop(X_train.index)
print(X_train)

etiquetas_entrenamiento = X_train.pop("Revenues")
estiquetas_test= X_test.pop("Revenues")

print(etiquetas_entrenamiento)

modelo= LinearRegression()
modelo.fit(X_train,etiquetas_entrenamiento)

predicciones = modelo.predict(X_test)
print(predicciones)

error=np.sqrt(mean_squared_error(estiquetas_test, predicciones))
print("Error porcentual: %f" % (error*100))

nueva_temperatura = pd.DataFrame(np.array([5]),columns=["Temperature"])
print(nueva_temperatura)

print(modelo.predict(nueva_temperatura ))