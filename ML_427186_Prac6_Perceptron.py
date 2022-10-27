# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:38:14 2022

@author: jorge
"""

#Se importan las Librerias
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
import matplotlib.patches as mpatches
#Se crea la clase perceptron
class Perceptron(object):
    
    def __init__(self,w0 = 0, w1=0.1, w2=0.1):
        # pesos
        self.w0 = w0    # bias
        self.w1 = w1
        self.w2 = w2
    #La funcion escalon     
    def step_function(self, z):
        if z >= 0:
            return 1
        else:
            return 0
    #funcion de suma de pesos    
    def weighted_sum_inputs(self, x1, x2):
        return sum([self.w0, x1 * self.w1, x2 * self.w2])
    
    #funcion de predeccion
    def predict(self, x1, x2):
        #Usa la funcion escalon para determinar la salida
        z = self.weighted_sum_inputs(x1, x2)
        return self.step_function(z)
    
    #funcion de  hiperplano
    def predict_boundary(self, x):
        return -(self.w1 * x + self.w0) / self.w2
    
    
    def fit(self, X, y, epochs=1, step=0.1, verbose=True):
        # Se entrena el percpetron con el dataset
        errors = []
        for epoch in range(epochs):
            error = 0
            for i in range(0, len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                # El update es proporcional al tamaño del paso y al error.
                update = step * (target - self.predict(x1, x2))
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print('Epochs: {} - Error: {} - Errors from all epochs: {}'
                      .format(epoch, error, errors))
   
#Creamos dataset
X, y_true = make_blobs(n_samples=40, centers=[(0, 0), (0, 1), (1, 0), (1,1)], cluster_std=0.1,random_state=15)
#Actualizamos las etiquetas de grupo
y = [1 if i== 3 else 0 for i in y_true]

x_train = pd.DataFrame(X)
y_train = pd.DataFrame(y)

#Creamos nuestra neurona 
my_perceptron = Perceptron(w1=0.2,w2=0.2)
#Entrenamos nuestra neurona, podemos cambiar la epoca dependiendo del data set al igual que el step.
my_perceptron.fit(x_train, y_train, epochs=4,step =0.1)

#Creamos vector para crear hiperplano
x = np.linspace(-0.2, 1.25)
y_predict = list()
y_2 = list()
count=40
for i in range(count):
    #Creamos vector para las etiquetas predichas
    y_predict.append(my_perceptron.predict(X[i,0],X[i,1]))


for i in x: 
    #Creamos la recta hiperplano
    y_2.append(my_perceptron.predict_boundary(i))

#Hacemos graficas
fig = plt.figure(figsize=(16,6),dpi =300)
fig.tight_layout()
ax = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)

#Grafica de dataset  
a=ax.scatter(X[:, 0], X[:, 1], c=y,cmap ='Paired',edgecolor ='k', s=50)
ax.set_title("Poblacion de AND")
legend1 = ax.legend(*a.legend_elements(), loc=8, title="grupos")
ax.add_artist(legend1)

#Grafica de dataset con hiperplano
ax1.plot(x,y_2,linestyle='--')
hiper = mpatches.Patch(color='b',label='Hiperplano')
c=ax1.legend(handles=[hiper])
b=ax1.scatter(X[:,0],X[:,1], c= y_predict,s=50,cmap ='Paired',edgecolor='k')
ax1.set_title("Poblacion clasificada con Hiperplano")
legend2 = ax1.legend(*b.legend_elements(),loc="upper left",title="grupos")
ax1.add_artist(c)
ax1.add_artist(legend2)
plt.show()

#Hacemos el repote de clasificacion:
print(f"\nReporte de Clasificacion : \n{classification_report(y_predict,y)}")
#Imprimimos Pesos ultimos:
pesos =[]
pesos.append(my_perceptron.w1)
pesos.append(my_perceptron.w2)
print('Los pesos finales son: \nw1={},w2={}'.format(pesos[0],pesos[1]))

#Creamos la matriz de confusion:
conf_matrix_perceptron = confusion_matrix(y, y_predict)
plt.figure(figsize=(10,8),dpi=200)
sns.set() 
sns.heatmap(conf_matrix_perceptron.T, cmap='Blues',annot=True, linewidths=0.5,fmt ='d', cbar=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Matriz de confusion')