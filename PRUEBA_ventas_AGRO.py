# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:12:23 2022

@author: andre
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


df=pd.read_excel("C:/Users/andre/Desktop/IA Python/Datos/Ventas_agro1.xlsx")

df=df.loc[:50]

df=df.drop(columns=["CLIENTE"])
df=df.drop(columns=["DESCRIPCION"])


#Primer llenado de nulos

miss_bool_marca = df["MARCA"].isnull()
miss_bool_marca
df.loc[miss_bool_marca, "MARCA"]=df.loc[miss_bool_marca, "MARCA"].apply(lambda x: "NEW HOLLAND")



#Segundo llenado de nulos

miss_bool_linea = df["LINEA"].isnull()
miss_bool_linea
df.loc[miss_bool_linea, "LINEA"]=df.loc[miss_bool_linea, "LINEA"].apply(lambda x: "PERNOS")

#Tercer llenado de nulos
miss_bool_unidad=df["UNIDAD"].isnull()
miss_bool_unidad
df.loc[miss_bool_unidad, "UNIDAD"] = df.loc[miss_bool_unidad, "UNIDAD"].apply(lambda x : "PZA")

#Borrar datos nulos
df=df[df["CODIGO"].notna()]
df=df[df["CANTIDAD"].notna()]
df=df[df["TOTAL"].notna()]

cod_uni=df.loc[:5,"CODIGO"]
total_cod=df.loc[10,"TOTAL"]

plt.figure(figsize=(10,7))
l_codigo=list(df["CODIGO"].unique())
x1=sns.barplot(data=df, x="CODIGO",y="TOTAL")
x1.set_xticklabels(labels=l_codigo,rotation=90)

plt.figure(figsize=(10,7))
l_codigo=list(df["CODIGO"].unique())
x1=sns.barplot(data=df, x="CODIGO",y="CANTIDAD")
x1.set_xticklabels(labels=l_codigo,rotation=90)




