# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:44:09 2022

@author: andre
"""


import sys
sys.path.append("C:/Users/andre/Desktop/IA Python/Sistema_prediccion/")
from Diabetes_project import * 

import tkinter
ventana = tkinter.Tk()
ventana.geometry("1000x3000")
ventana.title("Sistema de predicción")

titulo = tkinter.Label(ventana, text = "Sistema de predicción para pacientes con riesgo de diabetes",
                       font=("Arial",20))
titulo.pack()


lbl1 = tkinter.Label(ventana,text = "Numbers of pregnancies:" , font=("Arial",13))
lbl1.place(x=50, y=50, width = 200, height = 30)
txt1= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_preg)
txt1.place(x=270, y=50, width = 100, height=30)

lbl2 = tkinter.Label(ventana, text= "Glucose:", font=("Arial", 13))
lbl2.place(x=-10, y=100, width=200, height=30)
txt2= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_glu)
txt2.place(x=270, y=100, width = 100, height=30)

lbl3 = tkinter.Label(ventana, text= "Blood pressure:", font=("Arial", 13))
lbl3.place(x=12, y=150, width=200, height=30)
txt3= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_blood)
txt3.place(x=270, y=150, width = 100, height=30)

lbl4 = tkinter.Label(ventana,text = "Skin Thickness:" , font=("Arial",13))
lbl4.place(x=12, y=200, width = 200, height = 30)
txt4= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_skin)
txt4.place(x=270, y=200, width = 100, height=30)

lbl5 = tkinter.Label(ventana, text= "Insulin:", font=("Arial", 13))
lbl5.place(x=-20, y=250, width=200, height=30)
txt5= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_ins)
txt5.place(x=270, y=250, width = 100, height=30)

lbl6 = tkinter.Label(ventana, text= "BMI:", font=("Arial", 13))
lbl6.place(x=-25, y=300, width=200, height=30)
txt6= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_bmi)
txt6.place(x=270, y=300, width = 100, height=30)

lbl7 = tkinter.Label(ventana, text= "Diabetes pedigree functions:", font=("Arial", 13))
lbl7.place(x=10, y=350, width=300, height=30)
txt7= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_dpf)
txt7.place(x=270, y=350, width = 100, height=30)

lbl8 = tkinter.Label(ventana, text= "Age:", font=("Arial", 13))
lbl8.place(x=-25, y=400, width=200, height=30)
txt8= tkinter.Entry(ventana, bg="white",textvariable= var_predict.input_age)
txt8.place(x=270, y=400, width = 100, height=30)

btn1 = tkinter.Button(ventana, text= "Resultado", font=("Arial", 13),command=Pred_Diabetes.resultado)
btn1.place(x=30, y=550, width=170, height=30)

Res= tkinter.Label(ventana,bg="white",textvariable= Pred_Diabetes.resultado)
Res.place(x=270, y=550, width = 100, height=30)

ventana.mainloop()