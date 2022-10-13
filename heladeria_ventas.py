import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#importando datos
ventas_pre=pd.read_csv("C:/Users/andre/Desktop/IA Python/Datos/datos_ventas.csv")
 
#Visualizacion
sns.scatterplot(ventas_pre['Temperature'],ventas_pre['Revenues'] )

#Creando set de entrenamiento
X_train=ventas_pre['Temperature']
Y_train=ventas_pre['Revenues']

#Creando el modelo
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,input_shape=[1]))

model.summary()

#Compilado
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

#Entrenamiento
epochs_hist=model.fit(X_train,Y_train,epochs=1000)

keys=epochs_hist.history.keys()  #Tenemos como parametro la perdida

#Grafico de entrenamiento del modelo
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdida durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend(['Training loss'])


weights= model.get_weights() 

#Prediccion
temp=50
Revenue=model.predict([temp])
print("La ganancia segun la Red Neuronal sera: ", Revenue)


#Grafico de Prediccion
plt.scatter(X_train,Y_train, color= 'gray')  #Poner puntos de datos en color Gris
plt.plot(X_train,model.predict(X_train), color='red')    #Colocar linea roja
plt.ylabel('Ganancias[Dolares]')
plt.xlabel('Temperatura [gCelsius]')
plt.title('Ganancia Generada vs. Temperatura @Empresa de Helados')