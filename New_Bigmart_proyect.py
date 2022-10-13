# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:24:04 2021

@author: andre
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn import metrics 

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/train.csv")
print(df.info())
print(df.isnull().sum())

df["Item_Weight"]=df["Item_Weight"].fillna(df["Item_Weight"].mean())

outlet_size=df.pivot_table(values="Outlet_Size",columns="Outlet_Type",aggfunc=(lambda x:x.mode()[0]))
print(outlet_size)

missing_values=df["Outlet_Size"].isnull()

df.loc[missing_values,"Outlet_Size"]= df.loc[missing_values,"Outlet_Type"].apply(lambda x:outlet_size[x])

item_fat_content={"Item_Fat_Content":{
    "LF":"Low Fat",
    "reg":"Regular",
    "low fat":"Low Fat"}}

df.replace(item_fat_content,inplace=True)

sns.countplot(x="Item_Fat_Content", data=df)

plt.figure(figsize=(9,5))
sns.distplot(df["Item_MRP"])

plt.figure(figsize=(9,5))


l=list(df["Item_Type"].unique())
q=sns.countplot(df["Item_Type"])
q.set_xticklabels(labels=l,rotation=90)

l1=list(df["Outlet_Type"].unique())
q1=sns.countplot(x="Outlet_Type", data=df)
q1.set_xticklabels(labels=l1,rotation=75)


sns.distplot(df["Item_Outlet_Sales"])

### LABELENCODERS ###
encoder = LabelEncoder()
df["Item_Identifier"] = encoder.fit_transform(df["Item_Identifier"])
df["Item_Fat_Content"]= encoder.fit_transform(df["Item_Fat_Content"])
df["Item_Type"]= encoder.fit_transform(df["Item_Type"])
df["Outlet_Identifier"]= encoder.fit_transform(df["Outlet_Identifier"])
df["Outlet_Size"]= encoder.fit_transform(df["Outlet_Size"])
df["Outlet_Location_Type"]= encoder.fit_transform(df["Outlet_Location_Type"])
df["Outlet_Type"]= encoder.fit_transform(df["Outlet_Type"])

### Dividiendo los datos ###

X=df.drop(columns="Item_Outlet_Sales",axis=1)
y=df["Item_Outlet_Sales"]

X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.20,random_state=2)

xgbost=XGBRegressor()
xgbost.fit(X_train,y_train)

predict=xgbost.predict(X_train)

r2 = metrics.r2_score(y_train, predict)
print(r2)

error=np.sqrt(mean_squared_error(y_train, predict))
print(error*100)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

##### Data collection and Analysis #####
x=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/train.csv")
x
df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/train.csv")

print(df.head())

##### Number of rows and number of features #####
print(df.shape)

##### getting some informations about the dataset #####
print(df.info())

print(df.isnull().sum())

#Categorical Features ARE:
#Item_Identifier 
#Item_Fat_Content 
#Item_Type 
#Outlet_Identifier 
#Outlet_Size 
#Outlet_Location_Type 
#Outlet_Type

#####  Checking for missing values #####

#Mean value for Item_Weight
print(df["Item_Weight"].mean())

#Filling the missing values in "Item_Wright" column with "Mean value"
df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace =True)

#Replacing the missing values in "Outlet_Size" with the most repetied value
mode_outlet_size = df.pivot_table(values="Outlet_Size", columns ="Outlet_Type", aggfunc=(lambda x: x.mode()[0]))

print(mode_outlet_size)

missing_values=df["Outlet_Size"].isnull()
print(missing_values)


df.loc[missing_values, "Outlet_Size"] = df.loc[missing_values,"Outlet_Type"].apply(lambda x: mode_outlet_size[x])

#####  Data Analysis #####

print(df.describe())

#####  Numerical Features #####
#Item_Wright distribution
plt.figure(figsize=(9,5))
sns.distplot(df["Item_Weight"])


#Item_Visibility distribution
plt.figure(figsize=(9,5))
sns.distplot(df["Item_Visibility"])


#Item_MRP distribution
plt.figure(figsize=(9,5))
sns.distplot(df["Item_MRP"])

#Item_Outlet_Sales distribution
plt.figure(figsize=(9,5))
sns.distplot(df["Item_Outlet_Sales"])


#Item_Fat_Content column
plt.figure(figsize=(9,5))
sns.countplot(x="Item_Fat_Content",data=df)


#Item_Type column
l=list(df["Item_Type"].unique())

q=sns.countplot(df["Item_Type"])

q.set_xticklabels(labels=l,rotation=90)

#Outlet_Size column
plt.figure(figsize=(9,5))
sns.countplot(x="Outlet_Size",data=df)

##### Juntar las 3 variables LF ####
df["Item_Fat_Content"].value_counts()

df.replace({"Item_Fat_Content":
            {"LF":"Low Fat",
            "low fat":"Low Fat",
            "reg":"Regular"}},inplace=True)

### Label Encoder ###

#Item_Fat_Content 
#Item_Type 
#Outlet_Identifier 
#Outlet_Size 
#Outlet_Location_Type 
#Outlet_Type

encoder= LabelEncoder()
df["Item_Identifier"]=encoder.fit_transform(df["Item_Identifier"])
df["Item_Fat_Content"]=encoder.fit_transform(df["Item_Fat_Content"])
df["Item_Type"]=encoder.fit_transform(df["Item_Type"])
df["Outlet_Identifier"]=encoder.fit_transform(df["Outlet_Identifier"])
df["Outlet_Size"]=encoder.fit_transform(df["Outlet_Size"])
df["Outlet_Location_Type"]=encoder.fit_transform(df["Outlet_Location_Type"])
df["Outlet_Type"]=encoder.fit_transform(df["Outlet_Type"])


### Splitting features and Target ###
X = df.drop(columns="Item_Outlet_Sales", axis=1)
Y = df["Item_Outlet_Sales"]

### Splitting the data into Training data and Testing Data ###

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=2)

### Machine Learning Model Training ###

regressor = XGBRegressor()
regressor.fit(X_train,y_train)

### Prediction on training data ###
training_data_prediction = regressor.predict(X_train)

# R squared value

r2_train=metrics.r2_score(y_train,training_data_prediction)

print("R Squared value: ",r2_train )

# R squared value
test_data_prediction = regressor.predict(X_test)

r2_test=metrics.r2_score(y_test,test_data_prediction)

print("R Squared value: ",r2_test )

"""
