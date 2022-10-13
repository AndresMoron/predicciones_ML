# -*- coding: utf-8 -*-
"""
Created on Mon May 30 23:51:29 2022

@author: andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import cmath
import seaborn as sns
import os
import itertools
import glob
import shutil

df=pd.read_excel("C:/Users/andre/Desktop/IA Python/Datos/Date_Fruit_Datasets.xlsx")
df.describe

plt.figure(figsize=(24,12))
sns.heatmap(df.corr(),annot=False)

plt.figure(figsize=(16,8))
sns.scatterplot(x=df["AREA"], y=df["PERIMETER"], hue=df["Class"])

sns.countplot(data=df, x="Class",order=df["Class"].value_counts().sort_values().index)

sns.barplot(data=df, x="Class", y="AREA", order=df["Class"].value_counts().sort_values().index)

solidity = df["SOLIDITY"].sort_values()

solidity.head()
solidity.tail()

major_axis = df["MAJOR_AXIS"].sort_values()

major_axis.head()
major_axis.tail()

df=df[df["SOLIDITY"] > 0.93]
df=df[df["MAJOR_AXIS"]<1000]

Features = df.drop("Class",axis=1)
Label = df["Class"]

todrop = ['EXTENT','COMPACTNESS','StdDevRR','EntropyRG','KurtosisRB','ECCENTRICITY','SkewRR','MeanRB',
           'ASPECT_RATIO','SOLIDITY','SHAPEFACTOR_3','SHAPEFACTOR_4','KurtosisRR']

Features.drop(todrop,axis=1,inplace=True)

scaler = StandardScaler()
scaler.fit(Features)
scaled = scaler.transform(Features)

X=scaled
y=Label
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)












