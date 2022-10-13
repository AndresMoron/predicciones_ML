# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random 
from numpy import random
import datetime as dt
import seaborn as sns


tips=sns.load_dataset("tips")
sns.boxplot(data=tips,x="time", y= "tip", palette="husl",hue="smoker")
plt.xlabel("Dia")
plt.yblabel("Propina")




"""
titanic = sns.load_dataset("titanic")
titanic.head()

 
sns.displot(data=titanic,x="pclass",y="age",kind="kde"
            ,rug=True,hue="who",fill=True,binrange=5)

plt.title("Clase por pasajeros")

plt.show()





penguin=sns.load_dataset("penguins")
sns.histplot(data=penguin,x="body_mass_g", color ="c",
             cumulative=False, binwidth=500, binrange=[2500,7000],
             kde= True,hue="species", element="poly")
plt.title("Masa corporal de los pinguinos")
plt.xlabel("Masa corporal")




titanic = sns.load_dataset("titanic")
sns.barplot(data=titanic, x= "fare", y="class", hue="sex",
            palette="Spectral", order= ["Third","Second","First"],ci=None)
plt.title("Promedio pagado en funcion a la clase social")
plt.xlabel("Clase social")
plt.ylabel("Promedio pagado")





flights=sns.load_dataset("flights")
flights.head()

sns.lineplot(data=flights,x="year",y="passengers",color="green",
             hue="month",palette=sns.color_palette("hls",12))
plt.title("Total de pasageros por año")
plt.xlabel("Año")
plt.ylabel("Nro de pasageros")



flights_jan = flights.query("month == 'Jan'")
flights.head()

sns.lineplot(data=flights_jan,x="year",y="passengers",color="green")
plt.title("Total de pasageros por año")
plt.xlabel("Año")
plt.ylabel("Nro de pasageros")




tips=sns.load_dataset("tips")

sns.scatterplot(data=tips, x="tip",y="total_bill",color="purple",hue="smoker",palette=["red","blue"],
                style="smoker",size="size",sizes=[10,25,35,55,75,90])
plt.title("Propina vs todal pagado")
plt.xlabel("Propina")
plt.ylabel("Total pagado")




dates = ["1/9/2020","2/9/2020","3/9/2020","4/9/2020","5/9/2020",
         "6/9/2020","7/9/2020","8/9/2020","9/9/2020","10/9/2020",
         "11/9/2020","12/9/2020","13/9/2020","14/9/2020","15/9/2020",
         "16/9/2020","17/9/2020","18/9/2020","19/9/2020","20/9/2020",
         "21/9/2020","22/9/2020","23/9/2020","24/9/2020","25/9/2020",
         "26/9/2020","27/9/2020","28/9/2020","29/9/2020","30/9/2020",
        ]

x=[dt.datetime.strptime(d,"%d/%m/%Y").date() for d in dates]#Se extrae solo las fechas
y=np.random.randint(10000,20000,len(x))


plt.title("Total de ventas en Septiembre 2020")
plt.xlabel("Dias del mes")
plt.ylabel("Total ventas")
plt.xticks(rotation=90)
plt.plot(x,y,c="blue",ls="--",lw=2,marker="o")
plt.show()


data=np.random.randint(200,size=100)

plt.hist(data, bins=10, range=(0,200),color="#ff85b0",
         edgecolor="k",
         histtype="stepfilled"  #Tipo de barra
         )



options=["Juego del calamar","Breaking bad","Skins","Lucifer","Peaky","Rick"]
porcentaje=[70,15,5,10,5,5]


plt.pie(porcentaje,colors=["red","blue","m","green","#ff7f7d","pink"]
        ,labels=options,autopct="%1.0f%%",
        shadow=(True),
        labeldistance=(1.3), 
        startangle=80)
plt.title("Series mas vistas en el ultimo mes")


grados=["Reprobado","Aprobado","Notable","Excelente"]
count=[35,55,23,7]  #Las alturas / tabla de frecuencia
plt.barh(grados,count,color=["#ff7f7d","red","blue","green"],height=0.5,align="edge")
plt.xlabel("Notas")
plt.ylabel("Numero de alumnos")
plt.title("Notas de los alumnos de una clase")

x=random.randint(0,100,size=50)
y=random.randint(0,100,size=50)


plt.title("Estadistica de Numeros Random")
plt.xlabel("Numeros Eje X")
plt.ylabel("Numeros Eje Y")
plt.plot(y, "-g",x,"-.*r")
#plt.plot(x,c="blue",ls="--",lw=2,marker="*")
#plt.plot(y,c="red",ls="--",lw=1,marker="o")
plt.show()

num_infectados=[300000,120000,40000,750000,100000]
paises=["EEUU","Alemania","Argentina","Rusia","Mexico"]

plt.title("Pasies mas infectados Gestion 2020")
plt.xlabel("Paises")
plt.ylabel("numero infectados")

plt.plot(paises, num_infectados,c="red",ls="--",lw=2,marker=("*"),
         ms=20,alpha=0.5,mfc="m")

x=[1,2,3,4,5,6,7]
color=[33,22,55,66,45,84,1]
#plasma_r  es el reverso de los colores
plt.scatter(x=x, y=x, c=color,cmap="plasma_r",marker="o",s=200,alpha=1)
plt.title("Color plasma")
plt.colorbar()
print(plt.show)"""


"""
height=[174.3,155.5,162.1,157.9,174.8,169.2,172.0]
weight=[65.7,59.2,57.3,61.5,77.3,90.7,85.4]

plt.title("Altura vs Pesos")
plt.xlabel("Altura (cm)")
plt.ylabel("Pesos (kg)")
plt.scatter(x=height,y=weight,
            c="black",edgecolors="green",alpha=1,
            marker="v", s=200,linewidths=2)

print(plt.show())


height_boy=[174.3,155.5,162.1,157.9,174.8,169.2,172.0]
weight_boy=[65.7,59.2,57.3,61.5,77.3,90.7,85.4]

height_girl=[155.3,175.5,100.1,114.9,111.8,177.2,120.0]
weight_girl=[55.7,79.2,37.3,61.5,57.3,40.7,65.4]

plt.title("Altura chicos vs chicas")
plt.xlabel("Alturas(cm)")
plt.ylabel("Pesos (kg)")
plt.scatter(x=height_boy, y=weight_boy, c="red", marker="x", s=100)
plt.scatter(x=height_girl, y=weight_girl, c="blue", marker="o", s=100)"""