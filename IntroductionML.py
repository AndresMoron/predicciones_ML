# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:26:04 2021

@author: andre
"""
#ALGORITMO DE ARBOLES DE DECISIONES
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

df=datasets.load_boston()

print(df.feature_names)
print(df.keys())

#### PREPARAMOS LA DATA DE ARBOLES ####

#Seleccionamos la columna 6 del dataset
X_adr=df.data[:,np.newaxis,5]

#Defino la variable independiente
y_adr=df.target

plt.scatter(X_adr,y_adr)

#Entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train=train_test_split(X_adr,y_adr,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

#Defino algoritmo a utilizar
adr=DecisionTreeRegressor(max_depth=5)

#Se entrena el modelo
adr.fit(X_train,y_train)

#Predict
y_pred=adr.predict(X_test)


#Graficamos el modelo predictivo
X_grid= np.arange(min(X_test), max(X_test), 0.1)
X_grid= X_grid.reshape((len(X_grid),1))

plt.scatter(X_test,y_test)
plt.plot(X_grid,adr.predict(X_grid),color="red",linewidth=3)
plt.show()

print(adr.score(X_train,y_train))













"""
#ALGORITMO SVR VECTORES
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

boston=datasets.load_boston()
print(boston)

#Informacion del dataset
print(boston.keys())

#Caracteristicas
print(boston.DESCR)

#Verifico la informacion de las columnas
print(boston.feature_names)

#Seleccionamos la columna 6 del dataset
x_svr= boston.data[:,np.newaxis,5]

#Definimos la etiqueta
y_svr=boston.target

#Graficamos
plt.scatter(x_svr,y_svr)

#### Implementacion de Vectores de soporte Regresion #####

#Separar los datos de pruebas y train para el algoritmo
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train=train_test_split(x_svr,y_svr,test_size=0.2)


from sklearn.svm import SVR

#Defino el algoritmo a usar
svr=SVR(kernel="linear",C=1.0, epsilon=0.2)

#Entrenamos el modelo
svr.fit(X_train,y_train)

#Realizamos prediccion
y_pred=svr.predict(X_test)

#Graficamos los datos con el modelo
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color="red",linewidth=3)










#ALGORITMO DE REGRESION POLINOMIAL
import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt


boston=datasets.load_boston()
print(boston.keys())
print("Nombres de columnas")
print(boston.feature_names)


X_p=boston.data[:,np.newaxis,5]
y_p=boston.target

plt.scatter(X_p, y_p)
plt.show()

#Dividimos los datos de "train" en entrenamiento y prueba para probar los algoritmos
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_p,y_p,test_size=0.2 )

#Se define el grado del polinomio
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree=2)

#Se transforma las caracteristicas existentes en caracteristicas de mayor grado
#Estamos calculando los valores de X
X_train_poli= poli_reg.fit_transform(X_train)
X_test_poli= poli_reg.fit_transform(X_test)

#Defino el algoritmo a utilizar
pr=linear_model.LinearRegression()
pr.fit(X_train_poli,y_train)

#Realizo la prediccion
y_pred_pr=pr.predict(X_test_poli)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test) #Se grafican los TEST
plt.plot(X_test, y_pred_pr,color="red",lw=3) #PLOT de TEST y PRED

print("Valor de la pendiente o coeficiente 'a'")
print(pr.coef_)

print("Valor de la interseccion o coeficiente 'b'")
print(pr.intercept_)

print("Precision del modelo")
print(pr.score(X_train_poli, y_train))

###################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
print(boston.keys())

print("Nombre de las columnas de feature_names")
print(boston.feature_names)

print(boston)
######## PREPARAR LA DATA REGRESION LINEAL MULTIPLE #########

#Seleccionamos las columnas 5, 6 y 7 del dataset
X_multiple=boston.data[:,5:8]
print(X_multiple)


#Se define los datos correspondientes a las etiquetas
y_multiple=boston.target

######## IMPLEMENTACION DE REGRESION LINEAL MULTIPLE #########

#Se separan los datos de "Train" en entrenamiento y prueba para probar los algoritmos
X_train,X_test,y_train,y_test = train_test_split(X_multiple,y_multiple, test_size=0.2)

#Definimos el algoritmo a utilizar
lr_multiple=linear_model.LinearRegression() #Se usa la misma libreria

#Entrenamiento del modelo
lr_multiple.fit(X_train,y_train)

#Realizar prediccion
Y_pred_multiple= lr_multiple.predict(X_test)

print("DATPS DEL MODELO REGRESION LINEAL MULTIPLE\n")

print("Valor de las pendientes o coeficientes 'a': \n")
print(lr_multiple.coef_)

print("Valor de las interseccion o coeficientes 'b': \n")
print(lr_multiple.intercept_)

print("Precision del modelo: \n")
print(lr_multiple.score(X_train,y_train))



###########################################################

a=[0,1,2,3,4,5]
bins=[0,1,2,30,10,40]
plt.pie(a,bins )
plt.show()"""




"""
#Practica Regresion lineal Simple
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


#Cargar dataset
boston=datasets.load_boston()
#Imprime los datos principales del dataset
print(boston.keys())

#Cantidad de datos que tiene el dataset
print("Cantidad de datos: ", boston.data.shape)

#Etiquetas de cada columna
print("Nombre de columnas: ", boston.feature_names)

#Se debe seleccionar solo un dato para realizar la prediccion
#Seleccionamos la columna 5
X=boston.data[:, np.newaxis,5]

#Se define los datos correspondientes a la etiquetas
Y=boston.target

#Graficamos los datos con dispersion
plt.scatter(X, Y)
plt.xlabel("Numero de habitaciones")
plt.ylabel("Valor promedio")

#Se separan los datos de "Train" en entrenamiento y prueba para probar los algoritmos
#
X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

#Se define el algoritmo a utilizar
lr = linear_model.LinearRegression()

#Entrenamos el modelo
lr.fit(X_train, y_train)

#Realizando la prediccion
Y_pred=lr.predict(X_test)

#Graficando los datos junto con el modelo
plt.scatter(X_test,y_test)
plt.plot(X_test,Y_pred,color="red",linewidth=3)
plt.title("Regresion lineal Simple")
plt.xlabel("Numero de habitaciones")
plt.ylabel("Valor medio")

print("Valor de la pendiente o coeficiente a: ",lr.coef_ )

print("Valor de la interseccion o coeficiente b: ",lr.intercept_ )

#Calculamos la precision del logaritmo
#Devuelve el resultado de r al cuadrado
print("Precision del modelo: ", lr.score(X_train,y_train))
"""










