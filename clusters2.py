# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:21:40 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/segmentation data.csv")

df.info()

df.describe()

df_var = df.drop(columns=["ID"])

scaler= StandardScaler()
scaler.fit_transform(df_var)
df_scaler = scaler.fit_transform(df_var)


#Busqueda optima de clusters

wcss = []

for i in range (1,11):
    kmeans=KMeans(n_clusters=i,max_iter=300)
    kmeans.fit(df_scaler)
    wcss.append(kmeans.inertia_)
    
#Graficando los resultados de codo de jambu

plt.plot(range(1,11),wcss)
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS")

#Realizando la clusterizacion aplicando KMEANS

clustering = KMeans(n_clusters=4,max_iter=300)
clustering.fit(df_scaler)
df["KMEANS_Cluster"] = clustering.labels_

#Visualizando los clusters que se formaron

from sklearn.decomposition import PCA

pca= PCA(n_components=2)
pca_df = pca.fit_transform(df_scaler)
pca_p_df = pd.DataFrame(data=pca_df,columns=["Componente_1", "Componente_2"])
pca_nombres_p = pd.concat([pca_p_df, df[["KMEANS_Cluster"]]], axis=1)



fig= plt.figure(figsize=(6,6))

ax= fig.add_subplot(1,1,1)
ax.set_xlabel("Componente 1", fontsize=15)
ax.set_ylabel("Componente 2", fontsize=15)
ax.set_title("Componentes Principales", fontsize=20)

color_theme = np.array(["blue", "red","black","orange"])
ax.scatter(x=pca_nombres_p.Componente_1, y= pca_nombres_p.Componente_2,
           c=color_theme[pca_nombres_p.KMEANS_Cluster],s=50 )







