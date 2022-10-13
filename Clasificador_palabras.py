# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:24:52 2021

@author: andre
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Eopinions.csv")
print(df)

print(df["text"].str.split().str.len().describe())


train,test=train_test_split(df,test_size=0.33,random_state=42)

X_train=train["text"].to_list()
y_train=train["class"].to_list()

X_test=test["text"].to_list()
y_test=test["class"].to_list()


from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

X_train_vector=vectorizer.fit_transform(X_train)

X_test_vector=vectorizer.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()

clf_dec.fit(X_train_vector,y_train)

print(clf_dec.predict(X_test_vector[6]))

#Que tan bueno es el modelo
print(clf_dec.score(X_test_vector, y_test))

from sklearn.metrics import f1_score

print(f1_score(y_test,clf_dec.predict(X_test_vector),average=None))

test_prueba=["I like my car","That's camera is amazing"]

new_test=vectorizer.transform(test_prueba)

print(clf_dec.predict(new_test))