# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 23:57:40 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

columnas=["age","workclass","fnlwgt","education","education-num",
          "marital-status","occupation","relationship","race","sex",
          "capital-gain","capital-loss","hours-per-week","native-country","target"]


df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/adult.csv")

df.columns=columnas

print(df.info())
print(df["native-country"].unique())

target_data={"target":{
    " <=50K":0,
    " >50K":1}}

df.replace(target_data,inplace=True)

df["hours-per-week"].value_counts()

df["m-i-40"]=np.where(df["hours-per-week"]<=40,1,0)
 
print(df.head())
 
df["dif-capital"]=df["capital-gain"]-df["capital-loss"]

sex_data={"sex":{
    " Female":1,
    " Male":0}}

df.replace(sex_data,inplace=True)

df["usa"]=np.where(df["native-country"]==" United-States",1,0)

cat_col=["workclass","education",
          "marital-status","occupation","relationship","race",
          "native-country"]

for cat in cat_col:
    df[cat]=df.groupby(cat)[cat].transform("count")

X=df.drop(columns=["target"])
y=df["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=2)
print(X_train.shape, X_test.shape)
X_valid,X_test,y_valid,y_test = train_test_split(X_test,y_test,test_size=0.5,random_state=2)

print(X_valid.shape, X_test.shape)

xgb=XGBClassifier()

parameters= {"nthreads":[1],
             "objetive":["binary:logistic"],
             "learning_rate":[0.05,0.1],
             "n_estimators":[100,200]}

 #Validacion cruzada con todos los parametros para decirnos cual es la mejor combinacion
from sklearn.model_selection import GridSearchCV

#Nuevos parametros de entrenamiendo  

fit_params={"early_stopping_rounds":10, 
            "eval_metric":"logloss",
            "eval_set":[(X_test,y_test)]} #Si no mejora en las 10 rondas se detiene el entrenamiento

clf = GridSearchCV(xgb, parameters, cv=3,
                   scoring="accuracy")

clf.fit(X_train,y_train,**fit_params)

clf.best_estimator_
clf.best_score_

from sklearn.metrics import accuracy_score
best_xgb= clf.best_estimator_
y_preds=best_xgb.predict(X_valid)

comp = pd.DataFrame({"real":y_valid, "preds":y_preds})

acc= accuracy_score(y_valid, y_preds)
print(acc)












