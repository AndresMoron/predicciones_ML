# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:57:38 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/airfoil_self_noise.dat",sep="\t", header=None)

print(df.info())
print(df.shape)
columnas=["freq","angle","chord","velocty","suction","sound"]
df.columns=columnas

print(df.shape)
df.isnull().sum()

X=df.drop(columns="sound")
y=df["sound"]
print(X.shape)

### Graficar como se comporta la variable Y o predictora
eje_x = list(range(X.shape[0]))
eje_y = y

fig, ax = plt.subplots()
ax.plot(eje_x,eje_y,"-")
fig.set_size_inches(15,7)

import seaborn as sns
sns.set(style="ticks")
sns.pairplot(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

model=LinearRegression()
model.fit(X_train,y_train)

y_preds= model.predict(X_test)

comp=pd.DataFrame({"Valor_real:":y_test, "Valor_pred":y_preds })

### Evaluacion del MODELO ###
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, y_preds)
print(MSE)

#Grafica con el Y predict
eje_x = list(range(X_test.shape[0]))
eje_y = y_test
eje_y2 = y_preds

fig, ax = plt.subplots()
ax.plot(eje_x,eje_y,"-",eje_x,eje_y2,"o")
fig.set_size_inches(15,7)

### Representar los primeros 30 ###

def plot_20(y_ver,y_pre):
    eje_x = list(range(30))
    eje_y = y_test[:30]
    eje_y2 = y_preds[:30]

    fig, ax = plt.subplots()
    ax.plot(eje_x,eje_y,"-",eje_x,eje_y2,"o")
    fig.set_size_inches(15,7)

plot_20(y_test,y_preds)

### MODEL RIDGE ###

from sklearn.linear_model import Ridge
ridge= Ridge(alpha=0.1)
ridge.fit(X_train,y_train)

y_preds=ridge.predict(X_test)
print(mean_squared_error(y_test,y_preds))

alphas=np.arange(0.001,0.1,0.001)

mse=list()
for i in alphas:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train,y_train)
    y_preds=ridge.predict(X_test)
    mse.append(round(mean_squared_error(y_test,y_preds),3))
    
 values= pd.DataFrame({"alpha ":alphas, "mse": mse})
    
print(values[values["mse"]==values["mse"].min()])



### MODEL LASO ###
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.1)
lasso.fit(X_train,y_train)

y_preds=lasso.predict(X_test)
print(mean_squared_error(y_test,y_preds))

alphas1=np.arange(0.001,0.1,0.001)

mse=list()
for i in alphas1:
    lasso=Lasso(alpha=i)
    lasso.fit(X_train,y_train)
    y_preds=lasso.predict(X_test)
    mse.append(round(mean_squared_error(y_test,y_preds),3))
    
values_laso = pd.DataFrame({"alpha":alphas1,"mse":mse})
print(values_laso[values_laso["mse"]==values_laso["mse"].min()])
    










######### MY MODEL ############



"""
X=df.drop(columns="sound")
y=df["sound"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

model=LinearRegression()
model.fit(X_train,y_train)

com=pd.DataFrame({"real":y_test,"pred":y_train})



accuracy_test= model.predict(X_test)
print("Accuracy Test: ",metrics.r2_score(y_test, accuracy_test))

accuracy_train= model.predict(X_train)
print("Accuracy Test: ",metrics.r2_score(y_train, accuracy_train))

MSE=mean_squared_error(y_test, accuracy_test)
print("Accuracy Test: ",MSE)



com=pd.DataFrame({"real":y_test,"pred":accuracy_test})

input_data=(6300,12.3,0.1016,39.6,0.0408268)

input_data_np=np.asarray(input_data)

input_data_np_reshape=input_data_np.reshape(1,-1)


predict= model.predict(input_data_np_reshape)

print("Predict: ", predict)


plt.plot(y_test,color="red",label="Datos reales")
plt.plot(accuracy_test,color="blue",label="Datos predicction")

"""









