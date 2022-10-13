# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 09:24:42 2022

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Mall_Customers.csv")
print(df.info())
print(df.isnull().sum())

gender_value={"Gender":{"Male":0,"Female":1}}
df.replace(gender_value,inplace=True)

group=df.groupby(df["Gender"]).mean()

X=df.iloc[:,[3,4]].values
print(X)

wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
    
#plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title("The Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")


#Optimum Number of Clusters = 5

#Training the k-means clustering model

kmeans = KMeans(n_clusters=5,init="k-means++",random_state=0)

#Return a lavel for each data point based on their cluster
y= kmeans.fit_predict(X)
print(y)

#Visualizing all the Clusters
# 5 Clusters - 0,1,2,3,4

plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c="green",label="Cluster 1")
plt.scatter(X[y==1,0],X[y==1,1],s=50,c="red",label="Cluster 2")
plt.scatter(X[y==2,0],X[y==2,1],s=50,c="yellow",label="Cluster 3")
plt.scatter(X[y==3,0],X[y==3,1],s=50,c="violet",label="Cluster 4")
plt.scatter(X[y==4,0],X[y==4,1],s=50,c="blue",label="Cluster 5")

#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="black",label="Centroids")

plt.title("Customer Groups")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.ylabel("Annual Income")









