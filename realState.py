import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df =pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/Real estate.csv")

df2 = df.drop(columns=["X1 transaction date","No"],axis=1)

df2.hist(figsize=(10,10))

corr = df2.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, ax=ax)

print(df2.isnull().sum())
sns.regplot(x=df2["X3 distance to the nearest MRT station"],y=df2["X2 house age"])
sns.regplot(x=df2["X6 longitude"],y=df2["Y house price of unit area"])

X=df2.loc[:,"X2 house age":"X6 longitude"]
y=df2["Y house price of unit area"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X_train)

X_train_sca = scaler.transform(X_train)
X_test_sca = scaler.transform(X_test)

xx = np.arange(len(X_train_sca))
yy1 = X_train_norm[:,0]
yy2 = X_train_sca[:,0]
plt.scatter(xx,yy1,color="b")
plt.scatter(xx,yy2,color="r")

from sklearn.neighbors import KNeighborsRegressor as knn

model = knn(n_neighbors=3,p=1,algorithm="brute")
model.fit(X_train_norm,y_train)

ypred = model.predict(X_test_norm)

model.score(X_test_norm,y_test)

k_values = np.arange(1,100,2)

train_score_arr= []
val_score_arr = []
for k in k_values:
    model2 = knn(n_neighbors=k,p=1)
    model2.fit(X_train_norm,y_train)
    train_score= model.score(X_train_norm,y_train)
    train_score_arr.append(train_score*100)
    val_score = model2.score(X_test_norm,y_test)
    val_score_arr.append(val_score*100)
    print("k=%d, train_accuraccy=%.2f%%, test_accuracy=%.2f%%" % (k,train_score*100,
                                                                  val_score*100))
    
    
    
plt.plot(k_values,train_score_arr,"g")
plt.plot(k_values,val_score_arr,"r")
    
from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(model,X_train_norm,y_train,cv=10,scoring="r2")
cross_val_score_train.mean()

c=pd.DataFrame(ypred,columns=["Estimated Price"])
c.head()

d = pd.DataFrame(y_test)
d=y_test.reset_index(drop=True)

ynew = pd.concat([c,d],axis=1)





