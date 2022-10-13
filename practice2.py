# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 00:01:39 2022

@author: andre
"""
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/cancer.csv")

print(df.isnull().sum())
print(df.info())

sns.displot(df["diagnosis"])
sns.displot(df["radius_mean"])


X=df.drop(columns=["id","diagnosis","Unnamed: 32"])
y=df["diagnosis"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#Logistic Model

logistic=LogisticRegression()
logistic.fit(X_train,y_train)

test_train_acc = logistic.predict(X_train)
test_train= accuracy_score(test_train_acc,y_train)
print("Accuracy for train is: ",test_train)

test_test_acc = logistic.predict(X_test)
test_test= accuracy_score(test_test_acc,y_test)
print("Accuracy for train is: ",test_test)

#Vector Model

vector=svm.SVC(kernel="linear")
vector.fit(X_train,y_train)

test_train_vec = vector.predict(X_train)
test_train_vec= accuracy_score(test_train_vec,y_train)
print("Accuracy for train is: ",test_train_vec)

test_test_vec = vector.predict(X_test)
test_test_vec= accuracy_score(test_test_vec,y_test)
print("Accuracy for train is: ",test_test_vec)


#Tree Model

tree= DecisionTreeClassifier()
tree.fit(X_train,y_train )

test_train_tree = tree.predict(X_train)
test_train_tree= accuracy_score(test_train_tree,y_train)
print("Accuracy for train is: ",test_train_tree)

test_test_tree = tree.predict(X_test)
test_test_tree= accuracy_score(test_test_tree,y_test)
print("Accuracy for train is: ",test_test_tree)

input_data=(11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

input_data_np=np.asarray(input_data)

input_data_np_re=input_data_np.reshape(1,-1)

prediccion= tree.predict(input_data_np_re)

print("Prediccion: ",prediccion )
"""
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

df= pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/diabetes.csv")
print(df.info())
print(df.describe)

sns.distplot(df["Outcome"])
sns.distplot(df["Age"])
sns.distplot(df["Glucose"])

X=df.drop(columns="Outcome")
y=df["Outcome"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#Model
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

train_test_t= tree.predict(X_train)
accuracy_train = accuracy_score(train_test_t,y_train)
print("Accuracy for Train: ", accuracy_train)

test_test_t= tree.predict(X_test)
accuracy_test=accuracy_score(test_test_t,y_test)
print("Accuracy for Test: ",accuracy_test )

input_data=(6,148,72,35,0,33.6,0.627,50)
input_data_n=np.asarray(input_data)
input_data_n_r=input_data_n.reshape(1,-1)

prediccion = tree.predict(input_data_n_r)

if prediccion == 1:
    print("Tiene diabetes F")
else:
    print("Mas bien no")
    
print("Prediccion es: ", prediccion)
"""
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Copy_of_sonar_data.csv", header=None)
print(df.info())
df.describe()

df[60].value_counts()
sns.displot(df["R"])

R_data={"R":{"R":0,"M":1}}
df.replace(R_data,inplace=True)

X=


y=df["R"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)

#Model 1

tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

test_train_t=tree.predict(X_train)
acc_train=accuracy_score(test_train_t, y_train)
print("Accuracy for Train: ", acc_train)


test_test_t=tree.predict(X_test)
acc_test=accuracy_score(test_test_t, y_test)
print("Accuracy for Train: ", acc_test)

#Model 2

vector=svm.SVC(kernel="linear")
vector.fit(X_train,y_train)

test_train_t=vector.predict(X_train)
acc_train=accuracy_score(test_train_t, y_train)
print("Accuracy for Train: ", acc_train)


test_test_t=vector.predict(X_test)
acc_test=accuracy_score(test_test_t, y_test)
print("Accuracy for Train: ", acc_test)

#Model 3

logistic=LogisticRegression()
logistic.fit(X_train,y_train)

test_train_t=logistic.predict(X_train)
acc_train=accuracy_score(test_train_t, y_train)
print("Accuracy for Train: ", acc_train)


test_test_t=logistic.predict(X_test)
acc_test=accuracy_score(test_test_t, y_test)
print("Accuracy for Train: ", acc_test)

input_data=(0.0408,0.0653,0.0397,0.0604,0.0496,0.1817,0.1178,0.1024,0.0583,0.2176,0.2459,0.3332,0.3087,0.2613,0.3232,0.3731,0.4203,0.5364,0.7062,0.8196,0.8835,0.8299,0.7609,0.7605,0.8367,0.8905,0.7652,0.5897,0.3037,0.0823,0.2787,0.7241,0.8032,0.8050,0.7676,0.7468,0.6253,0.1730,0.2916,0.5003,0.5220,0.4824,0.4004,0.3877,0.1651,0.0442,0.0663,0.0418,0.0475,0.0235,0.0066,0.0062,0.0129,0.0184,0.0069,0.0198,0.0199,0.0102,0.0070,0.0055)
input_data_n=np.asarray(input_data)
input_data_n_r=input_data_n.reshape(1,-1)

prediccion = logistic.predict(input_data_n_r)
print(prediccion)"""

#Import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#Data Collection and Analysis
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Mall_Customers.csv")

print(df.describe())
print(df.shape)

#Getting some information about the dataset
print(df.info())

#Choosing the Annual Income Column And Spending Score Column
X= df.iloc[:,[3,4]].values

#Choosing the number of clusters
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #Nos da el resultado del WCSS
    
#Plot an elbow graph

sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Optimun number of clusters = 5

#Training the k-means Clustering Model

kmeans = KMeans(n_clusters=5,init="k-means++", random_state=0)

#Return a label for each data point based on the their cluster
y = kmeans.fit_predict(X)
print(y)

#Visualizing all the clusters

#Plotting all the clusters and their Centroids
plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0],X[y==0,1],s=50, c = "green", label="Cluster 1")
plt.scatter(X[y==1,0],X[y==1,1],s=50, c = "blue", label="Cluster 2")
plt.scatter(X[y==2,0],X[y==2,1],s=50, c = "yellow", label="Cluster 3")
plt.scatter(X[y==3,0],X[y==3,1],s=50, c = "gray", label="Cluster 4")
plt.scatter(X[y==4,0],X[y==4,1],s=50, c = "violet", label="Cluster 5")

#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c="black", label="Centroid")
plt.title("Customer groups")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")





















































