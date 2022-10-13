# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:56:17 2021

@author: andre
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
%matplotlib inline
warnings.filterwarnings("ignore")

df=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/train.csv")

print(df.info())
print(df.isnull().sum())
print(df.apply(lambda x: len(x.unique())))

#check for categorical attributes

cat_obj=[]
for x in df.dtypes.index:
    if df.dtypes[x] == "object":
        cat_obj.append(x)
print(cat_obj)

#We remove the names that not matter of dataset
cat_obj.remove("Item_Identifier")
cat_obj.remove("Outlet_Identifier")

for i in cat_obj:
    print(i)

for col in cat_obj:
    print(col)
    print(df[col].value_counts())
    print()
    
#Fill the missing values
item_weight_mean= df.pivot_table(values= "Item_Weight", index="Item_Identifier")
print(item_weight_mean)

miss_bool= df["Item_Weight"].isnull()
print(miss_bool)

for i, item in enumerate(df["Item_Identifier"]):
    if miss_bool[i]:
        if item in item_weight_mean:
            df["Item_Weight"][i] = item_weight_mean.loc[item]["Item_Weight"]
        else:
             df["Item_Weight"][i]=np.mean(df["Item_Weight"])


#Fill the values with "Small"
outlet_size_mode = df.pivot_table(values= "Outlet_Size", columns="Outlet_Type",aggfunc=(lambda x: x.mode()[0]))
print(outlet_size_mode)

miss_bool = df["Outlet_Size"].isnull()
df.loc[miss_bool, "Outlet_Size"] = df.loc[miss_bool, "Outlet_Type"].apply(lambda x:outlet_size_mode[x] )


#Replace one column "Item_Visibility" with the mean
df.loc[:, "Item_Visibility"].replace([0],[df["Item_Visibility"].mean()],inplace =True)

#Combine item fat content
df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"LF": "Low Fat",
                                                          "reg":"Regular",
                                           "low fat":"Low Fat"})

df["New_Item_Type"] = df["Item_Identifier"].apply(lambda x: x[:2])
print(df["New_Item_Type"])


df["New_Item_Type"] = df["New_Item_Type"].replace({"FD":"Food",
                                                               "NC":"Non-Consumable",
                                                               "DR":"Drinks"})
print(df["New_Item_Type"])


df["Item_Identifier"] = df["New_Item_Type"]


df.loc[df["New_Item_Type"] == "Non-Consumable","Item_Fat_Content"] = "Non-Edible"

df["Outlet_Year"] = 2013 - df["Outlet_Establishment_Year"]
df["Outlet_Establishment_Year"] = 2013- df["Outlet_Year"]
############### VISUALIZACION DE DATOS  ############
sns.displot(df["Item_Weight"])

sns.displot(df["Item_Visibility"])

sns.displot(df["Item_Outlet_Sales"])

sns.displot(df["Item_MRP"])

#Log transformation to smallest numbers
df["Item_Outlet_Sales"] = np.log(1+df["Item_Outlet_Sales"])

sns.displot(df["Item_Outlet_Sales"])

sns.countplot(df["Item_Fat_Content"])
p
#plt.figure(figsize=(15,5))
l=list(df["Item_Type"].unique())

chart=sns.countplot(df["Item_Type"])

chart.set_xticklabels(labels=l , rotation=90)

sns.countplot(df["Outlet_Size"])

sns.countplot(df["Outlet_Establishment_Year"])

sns.countplot(df["Outlet_Location_Type"])

l=list(df["Outlet_Type"].unique())
z=sns.countplot(df["Outlet_Type"])
z.set_xticklabels(labels=l,rotation=75)

############### Correlation Matrix ##############
corr= df.corr()
sns.heatmap(corr,annot=True, cmap="coolwarm")

############### Label Encoding ##############
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

df["Outlet"] = le.fit_transform(df["Outlet_Identifier"])

cat_col = ["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type","New_Item_Type"]
for col in cat_col:
    df[col]=le.fit_transform(df[col])

############### One Hot encoding ##############

df= pd.get_dummies(df, columns=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","New_Item_Type"])

############## Input Split ##############
X=df.drop(columns=["Outlet_Year", "Outlet_Identifier","Item_Outlet_Sales","Item_Identifier"])
y=df["Item_Outlet_Sales"]

############## Model Training ##############
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def train(model,X,y):
    #Train the model
    model.fit(X,y)
    
    #Predict the training set 
    pred = model.predict(X)
    
    #Perform cross-validation
    cv_score = cross_val_score(model,X,y,scoring="neg_mean_squared_error")
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE",mean_squared_error(y,pred))
    print("CV Score:", cv_score)
    

####### Other Model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model= LinearRegression(normalize=(True))
train(model,X,y)
coef= pd.Series(model.coef_,X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

####### Other Model

model= Ridge(normalize=(True))
train(model,X,y)
coef= pd.Series(model.coef_,X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

####### Other Model

model= Lasso()
train(model,X,y)
coef= pd.Series(model.coef_,X.columns).sort_values()
coef.plot(kind="bar",title="Model Coefficients")

####### Other Model

from sklearn.tree import DecisionTreeRegressor
model= DecisionTreeRegressor()
train(model,X,y)
coef= pd.Series(model.feature_importances_,X.columns).sort_values(ascending=False)
coef.plot(kind="bar",title="Feature Importance")

####### Other Model

from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
train(model,X,y)
coef= pd.Series(model.feature_importances_,X.columns).sort_values(ascending=False)
coef.plot(kind="bar",title="Feature Importance")

####### Other Model

from sklearn.ensemble import ExtraTreesRegressor
model= ExtraTreesRegressor()
train(model,X,y)
coef= pd.Series(model.feature_importances_,X.columns).sort_values(ascending=False)
coef.plot(kind="bar",title="Feature Importance")



print(df.describe())





















