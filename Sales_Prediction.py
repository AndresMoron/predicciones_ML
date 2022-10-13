# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:19:28 2022

@author: andre
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import warnings
%matplotlib inline 
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Train.csv")

# ------- Statistic data information -------

df.describe()
df.info()

#Preprocesado de datos
#Revisar valores unicos
df.apply(lambda x: len(x.value_counts()))

#Revisar nulos
df.isnull().sum()

df.drop(columns=["Item_Identifier","Outlet_Identifier"])

# ----------- Rellenar datos nulos -----------

#Rellenado de Item_Weight
df["Item_Weight"].mean()
miss_item_w = df["Item_Weight"].isnull()
miss_item_w
df.loc[miss_item_w,"Item_Weight"] = df.loc[miss_item_w,"Item_Weight"].apply(lambda x: df["Item_Weight"].mean())

#Rellenado de Outlet_Size
df["Outlet_Size"].value_counts()
miss_outlet = df["Outlet_Size"].isnull()
df.loc[miss_outlet,"Outlet_Size"] = df.loc[miss_outlet,"Outlet_Size"].apply(lambda x: "Medium")

#Revisando Fat_content
df["Item_Fat_Content"].value_counts()
values_fat = {"Item_Fat_Content":{"low fat":"Low Fat", "LF": "Low Fat", "reg": "Regular"}}
df.replace(values_fat,inplace=True)

# ----------- Creacion de nuevos atributos -----------

#Extraemos las 2 primera letras
df["New_Item_Type"] = df["Item_Identifier"].apply(lambda x:x[:2])

New_Item_Type = {"New_Item_Type":{"FD":"Food","NC":"Non-Consumable","DR":"Drinks"}}
df.replace(New_Item_Type,inplace=True)

#Nuevo atributo
df.loc[df["New_Item_Type"]=="Non-Consumable","Item_Fat_Content"] = "Non-Edible"

#Create valores para el anho de establecimiento
df["Outlet_Years"]=2013 - df["Outlet_Establishment_Year"]


# ----------- Explorando los analisis de datos (Graficos) -----------

df.head()

sns.distplot(df["Item_Weight"])
sns.distplot(df["Item_Visibility"])
sns.distplot(df["Item_MRP"])
sns.distplot(df["Item_Outlet_Sales"])
#Normalizamos con log para Item_Outlet_Sales
df["Item_Outlet_Sales"] = np.log(1+df["Item_Outlet_Sales"])
sns.distplot(df["Item_Outlet_Sales"])
#-----------------------------------------------------------
sns.countplot(df["Item_Fat_Content"])
l_itemType= list(df["Item_Type"].unique())
x=sns.countplot(df["Item_Type"])
x.set_xticklabels(labels=l_itemType,rotation=90)
sns.countplot(df["Outlet_Establishment_Year"])
sns.countplot(df["Outlet_Size"])
sns.countplot(df["Outlet_Location_Type"])
sns.countplot(df["Outlet_Type"])

# ----------- Matriz de Correlacion -----------

corr= df.corr()
sns.heatmap(corr,annot=True, cmap="coolwarm")

# ----------- Label Encoding -----------

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Outlet"] = le.fit_transform(df["Outlet_Identifier"])

cat_col=["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type","New_Item_Type"]
for col in cat_col:
    df[col]=le.fit_transform(df[col])
    
# ----------- Onehot Encoding -----------

df = pd.get_dummies(df, columns=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","New_Item_Type"])
df.shape

# ----------- Train-Test Split -----------

X=df.drop(columns=["Outlet_Establishment_Year","Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"])
y=df["Item_Outlet_Sales"]

# ----------- Entrenamiento del Modelo -----------

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def train(model,X,y):
    #Train the model
    model.fit(X,y)
    
    #Predict the training set
    prediction = model.predict(X)
    
    #Use cross-validation
    cv_score = cross_val_score(model, X,y,scoring="neg_mean_squared_error",cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE: ", mean_squared_error(y, prediction))
    print("CV Score: ",cv_score)


from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression(normalize=True)
train(model,X,y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

model = Ridge(normalize=True)
train(model,X,y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

model = Lasso()
train(model,X,y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model,X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values()
coef.plot(kind="bar",title="Features Importance")