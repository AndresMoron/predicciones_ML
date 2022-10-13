# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:27:09 2022

@author: andre
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/diabetes.csv")

print(df.info())
print(df.describe())
print(df["Outcome"].value_counts())


X=df.drop(columns=["Outcome"])
y=df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)

#Logistic Regression
logistic=LogisticRegression()
logistic.fit(X_train, y_train)

test_accuracy = logistic.predict(X_test)
test_accuracy_score= accuracy_score(test_accuracy,y_test)
print("Test Accuracy for TEST: ",test_accuracy_score)

train_accuracy = logistic.predict(X_train)
train_accuracy_score = accuracy_score(train_accuracy, y_train)
print("Train Accuracy for TRAIN: ",train_accuracy_score )

#SVM
vector=SVC(kernel="linear")
vector.fit(X_train, y_train)

test_accuracy = vector.predict(X_test)
test_accuracy_score= accuracy_score(test_accuracy,y_test)
print("Test Accuracy for TEST: ",test_accuracy_score)

train_accuracy = vector.predict(X_train)
train_accuracy_score = accuracy_score(train_accuracy, y_train)
print("Train Accuracy for TRAIN: ",train_accuracy_score )

#Tree
tree=DecisionTreeClassifier()
tree.fit(X_train, y_train)

test_accuracy = tree.predict(X_test)
test_accuracy_score= accuracy_score(test_accuracy,y_test)
print("Test Accuracy for TEST: ",test_accuracy_score)

train_accuracy = tree.predict(X_train)
train_accuracy_score = accuracy_score(train_accuracy, y_train)
print("Train Accuracy for TRAIN: ",train_accuracy_score )

Pregnancies=10
Glucose= 108
BloodPressure=66
SkinThickness=0
Insulin=0
BMI=32.4
DiabetesPedigreeFunction=0.272
Age=57

input_data=(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
input_data_np=np.asarray(input_data)
input_data_np_res=input_data_np.reshape(1,-1)

predict=tree.predict(input_data_np_res)

print(predict)










































"""
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/diabetes.csv")

print(df.shape)
print(df.describe())
print(df["Outcome"].value_counts())

media_outcome=df.groupby("Outcome").mean()

#Separating the data and labels
X=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]


### Data Standaritation ###

scaler = StandardScaler()
standarized_data=scaler.fit_transform(X)

X=standarized_data
y=df["Outcome"]


### Train Split ###

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, stratify=y,random_state=2)

### Model Training ###

classifier= svm.SVC(kernel="linear")
classifier.fit(X_train,y_train)


logistic = LogisticRegression()
logistic.fit(X_train, y_train)
### Model Evaluation ###
### Accuracy Score ###

X_train_prediction = classifier.predict(X_train)    #Se guarda en una variable las predicciones
training_data_accuracy = accuracy_score(X_train_prediction, y_train )  #Se compara con las etiquetas originales

print("Accuracy Score of the Training Data :", training_data_accuracy)


X_test_prediction = classifier.predict(X_test)    #Se guarda en una variable las predicciones
test_data_accuracy = accuracy_score(X_test_prediction, y_test )  #Se compara con las etiquetas originales

print("Accuracy Score of the Test Data :", test_data_accuracy)

### Making a Predictive System ###

input_data = (5,116,74,0,0,25.6,0.201,30)

#Changing the input_data to numpy array
input_data_as_np = np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_np.reshape(1,-1)

#Standarize the input data
std_data = scaler.transform(input_data_reshaped)

print(std_data)

prediction= classifier.predict(std_data)

print("Prediccion: ",prediction)

if prediction[0]   ==0: #Se pone 0 solo porque se quiere el primer valor y ademas es una lista
    print("The person is not diabetic")
else:
    print("The person is diabetic")

"""



















"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/diabetes.csv")

print(df.info())
print(df.isnull().sum())

sns.distplot(df["Pregnancies"])
sns.distplot(df["Glucose"])
sns.distplot(df["BloodPressure"])
sns.distplot(df["Age"])

X=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]

X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.20,random_state=0)

model=LinearRegression()
model.fit(X_train,y_train)

predict=model.predict(X_test)

prediccion=pd.DataFrame(np.array([[6,148,72,35,0,33.6,0.627,50]]),columns=("Pregnancies","Glucose","BloodPressure",
                                                                            "SkinThickness","Insulin","BMI","DiabetesPedigreeFunction",
                                                                            "Age"))

x=model.predict(prediccion)
print(x)

error=np.sqrt(mean_squared_error(y_test,predict))
print(error*100)
"""