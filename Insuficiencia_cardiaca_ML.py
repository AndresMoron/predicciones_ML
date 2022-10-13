# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:00:13 2021

@author: andre
"""
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/heart.csv")
print(df.keys())

df=df.drop(columns=("Sex"))
df=df.drop(columns=("ChestPainType"))

print(df.shape)

X=df[["Cholesterol"]]

y=df[["HeartDisease"]]

plt.scatter(X,y)
plt.ylabel("Daño corazon")
plt.xlabel("Colesterol")


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lr=linear_model.LinearRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)



error=np.sqrt(mean_squared_error(y_test, y_pred))
print(error)



"""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#################### PREPARACION DE DATOS #################

#Conjunto de datos de prediccion de insuficiencia cardiaca
cardio= pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/heart.csv")
print(cardio.keys())

#Mas importante -> 1.Age  2. resting blood pressure
#3.colesterol sérico 4.resting electrocardiogram results
#5. exercise induced angina 6.maximum heart rate achieved


print(cardio.info())
print(cardio.hist())

#Se eliminan las columnas menos relevantes para la prediccion
cardio = cardio.drop(columns=("Sex"))
cardio = cardio.drop(columns=("ChestPainType"))
cardio = cardio.drop(columns=("FastingBS"))
cardio = cardio.drop(columns=("Oldpeak"))
cardio = cardio.drop(columns=("ST_Slope"))

############### REVISION DE DATOS ###########################

#Revisar si hay datos nulos
print(cardio.isna().sum())

#Transformar cadenas a datos numericos
#Revisar cuantos valores hay en cada columna
print(cardio["RestingECG"].unique())
print(cardio["ExerciseAngina"].unique())

#Reemplazar los datos
reemplazo_valor_R= {"RestingECG":{
    "Normal":1,
    "ST":2,
    "LVH":3
    }}

cardio.replace(reemplazo_valor_R,inplace=True)

reemplazo_valor_E = {"ExerciseAngina":{
    "N":0,
    "Y":1
    }}

cardio.replace(reemplazo_valor_E,inplace=True)

print(cardio)

############# Relacion/visualizacion de variables ###########

plt.scatter(x=cardio["Age"], y=cardio["Cholesterol"])
plt.title("Edad vs Colesterol")
plt.xlabel("Edad")
plt.ylabel("Colesterol")
plt.show()

#Eliminamos los valores que no son relevantes con un query
#Nos quedamos con los datos mayores a 0
cardio=cardio.query("Cholesterol > 0")

############# ENTRENAMIENTO DEL MODELO #####################
#Dividir los datos en un 80% para el etrenamiento 
datos_entrenamiento=cardio.sample(frac=0.8,random_state=0)
# 20% para el test
datos_test=cardio.drop(datos_entrenamiento.index)

print(datos_entrenamiento)

#Se separa la variable que queremos predecir
etiqueta_entrenamiento= datos_entrenamiento.pop("HeartDisease")
etiqueta_test=datos_test.pop("HeartDisease")

print(etiqueta_entrenamiento)

modelo = LinearRegression()
modelo.fit(datos_entrenamiento,etiqueta_entrenamiento)

prediccion = modelo.predict(datos_test)
print(prediccion)

#Revisar el porcentaje de error
error=np.sqrt(mean_squared_error(etiqueta_test, prediccion))
print("Error porcentual: %f" % (error*100))


nueva_persona=pd.DataFrame(np.array([[100,200,190,3,200,0]]),columns=("Age","RestingBP","Cholesterol","RestingECG","MaxHR","ExcersiceAngina"))
print(nueva_persona)
x=modelo.predict(nueva_persona)
print(x)
if x>0.5:
    print("La persona tiene insuficiencia cardiaca")
else:
    print("La persona no tiene insuficiencia cardiaca")"""
    













