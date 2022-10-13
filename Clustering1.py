# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:14:14 2022

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("C:/Users/andre/Desktop/files/Rocio-Chavez-youtube-Files-master/caracteristicas de vinos.csv")

df.info()

df.describe()

df_variables = df.drop(columns=["Vino"], axis=1)

def minmax_norm(df_input):
    return (df_variables-df_variables.min()) / (df_variables.max() - df_variables.min())

df_minmax = minmax_norm(df_variables)

#Busqueda de la cantidad optima de clusters

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,max_iter=300)
    kmeans.fit(df_minmax)
    wcss.append(kmeans.inertia_)
    
#Graficando los resultados de WCSS para formar el Codo de Jambu

plt.plot(range(1,11),wcss)
plt.title("Codo de Jambu")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS") #indicador de que tan similares son los individuos


#Aplicando el metodo K-MEANS a la bd

clustering = KMeans(n_clusters=3,max_iter=300)
clustering.fit(df_minmax)

df["KMeans_Clisters"] = clustering.labels_


#Visualizando los clusters que se formaron

from sklearn.decomposition import PCA

pca= PCA(n_components=2)
pca_df = pca.fit_transform(df_minmax)
pca_vinos_df = pd.DataFrame(data=pca_df,columns=["Componente_1", "Componente_2"])
pca_nombres_vinos = pd.concat([pca_vinos_df, df[["KMeans_Clisters"]]], axis=1)

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Compenente 1", fontsize=15)
ax.set_ylabel("Componente 2", fontsize=15)
ax.set_title("Componentes Principales", fontsize=20)

color_theme = np.array(["blue","green","orange"])
ax.scatter(x = pca_nombres_vinos.Componente_2, y = pca_nombres_vinos.Componente_1,
           c=color_theme[pca_nombres_vinos.KMeans_Clisters], s =50)









