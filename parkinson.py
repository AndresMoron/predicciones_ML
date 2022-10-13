

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/parkinsons.csv")
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.shape)

#Distribution of target variable
df["status"].value_counts()

group=df.groupby("status").mean()

#### DATA PRE-PROCESSING

#Separating the features Y Target
X=df.drop(columns=["name","status"],axis=1)
y=df["status"]

#Training data and Test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Data Standarization
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#### MODEL TRAINING

model = svm.SVC(kernel="linear")

#Training the svm model with training data
model.fit(X_train,y_train)

### MODEL EVALUATION 

#Accuracy score
X_train_prediction= model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)


print("Accuracy score of training data: ", training_data_accuracy)

### Acuraccy for test data

X_test_prediction= model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)


print("Accuracy score of test data: ", test_data_accuracy)

### BUILDING A PREDICTIVE SYSTEM

input_data= (243.43900,250.91200,232.43500,0.00210,0.000009,0.00109,0.00137,0.00327,0.01419,0.12600,0.00777,0.00898,0.01033,0.02330,0.00454,25.36800,0.438296,0.635285,-7.057869,0.091608,2.330716,0.091470)

input_data_as_np = np.asarray(input_data)

input_data_reshape = input_data_as_np.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

prediction = model.predict(std_data)

print("Prediction: ", prediction)

import pickle
with open("svm_parkinson.sav","wb") as f:
    pickle.dump(model,f)

with open("svm_parkinson.sav","rb") as f:
    loaded_model = pickle.load(f)
    
    


