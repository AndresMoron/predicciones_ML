# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:58:01 2021

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error

bitcoin = pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Bitcoin_SV.csv")
print(bitcoin.keys())
print(bitcoin.info())

bitcoin = bitcoin.drop(columns=["SNo"])
bitcoin = bitcoin.drop(columns=["Change %"])
bitcoin = bitcoin.drop(columns=["Open"])
bitcoin = bitcoin.drop(columns=["Vol."])
print(bitcoin.info())

bitcoin["Date"]=pd.to_datetime(bitcoin["Date"],errors = 'coerce',format = '%Y-%m-%d').dt.strftime("%Y%d%m")
print(bitcoin["Date"])
bitcoin["Date"]=pd.to_numeric(bitcoin["Date"])
print(bitcoin.info())
"""date=bitcoin["Date"]
x=date.strftime("%Y%m%d")
z=int(x)
print(z)
print(type(z))
"""

plt.scatter(x=bitcoin["Date"],y=bitcoin["Price"])
plt.xlabel("Fecha")
plt.ylabel("Precio")
#################################################

datos_entrenamiento= bitcoin.sample(frac=0.8, random_state=0)
datos_test=bitcoin.drop(datos_entrenamiento.index)

etiqueta_entrenamiento=datos_entrenamiento.pop("Date")
etiqueta_test=datos_test.pop("Date")

modelo=LinearRegression()
modelo.fit(datos_entrenamiento,etiqueta_entrenamiento)

prediccion=modelo.predict(datos_test)

error=np.sqrt(mean_squared_error(etiqueta_test, prediccion))
print("Error porcentual: %f" % (error*100))

nueva_fecha=pd.DataFrame(np.array([[100000,100500,80000]]),columns=("Price","High","Low"))
print(modelo.predict(nueva_fecha))






