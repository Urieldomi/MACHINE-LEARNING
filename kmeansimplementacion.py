# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:58:01 2022

@author: jorge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
#Leemos los datos de entrada para generar el dataset:
X = pd.read_csv('dataset1.csv')
x = np.array(X)
#Las variables
klusters=3
n_centros=3
centros=[]
matriz=[]

def distanciaeu(a,b):
  distancia = pow(b[0]-a[1],2) + pow(b[0]-a[1],2)
  return distancia
    
for i in range(n_centros):
    puntos_aleatorios=random.choice(x)
    centros.append(puntos_aleatorios)
    c=np.array(centros)

N=len(X)
iteraciones=10
for n in range(iteraciones):
    for i in range(N):
        distance_i = []
        for j in range(n_centros):
            distance_i.append(distanciaeu(c[j],x[i]))
            distance_i.clear()

minimo = distance_i    
    
    
    