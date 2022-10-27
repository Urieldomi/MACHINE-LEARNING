# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:54:25 2022

@author: jorge

"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#Leemos los datos de entrenamiento y prueba que contienen también las etiquetas del dataset
data = pd.read_csv('dataset_classifiers3.csv',header=0)

#Separamos los datos vectoriales de las etiquetas de clase
data_vectors  = data.loc[:,['h','p']]
data_vectors = np.array(data_vectors)

data_eti = data.loc[:,'y_true']
data_eti =np.array(data_eti)

#Creamos la partición de entrenamiento y prueba con una proporción 90/10.
X_train, X_test, y_train, y_test = train_test_split(data_vectors, data_eti, test_size = 0.10, shuffle = True, random_state = 2)

#Aplicamos el clasificador kNN
clf_knn = KNeighborsClassifier(n_neighbors = 15)
clf_knn.fit(X_train, y_train)
y_pred = clf_knn.predict(X_test)

plt.figure(figsize=(10,8),dpi=140)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40,cmap='Accent')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40,cmap='Paired', marker = 'X')
file = open("Reporte de Clasificacion.txt","w")
file.write("----------------------K-NN---------------------------------------\n"
f"Training Set Score : {clf_knn.score(X_train, y_train) * 100} %\n"
f"Test Set Score : {clf_knn.score(X_test, y_test) * 100} % \nModel Classification Report : \n{classification_report(y_test, clf_knn.predict(X_test))}")
file.write("--------------------------------------------------------------------\n\n")

print("------------------K-NN-----------------------------------------------")
print(f"Training Set Score : {clf_knn.score(X_train, y_train) * 100} %")
print(f"Test Set Score : {clf_knn.score(X_test, y_test) * 100} %")
 
# Printing classification report of classifier on the test set set data
print(f"\nModel Classification Report : \n{classification_report(y_test, clf_knn.predict(X_test))}")
conf_matrix_KNN = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8),dpi=100)
sns.set() 
sns.heatmap(conf_matrix_KNN.T, cmap='Blues',annot=True, linewidths=0.5,fmt ='d', cbar=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Matriz de confusion K-NN')
print("----------------------------------------------------------------------")

#plt.contourf(XX, YY, Z_pred,cmap = 'viridis', alpha = 0.4)

#Aplicamos como clasificador una SVM lineal
clf_svc = SVC(kernel = 'rbf')

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 5, 10, 50,100]}
grid = GridSearchCV(clf_svc, param_grid) 
grid.fit(X_train, y_train)
print(grid.best_params_)
clf_svc = grid.best_estimator_

clf_svc.fit(X_train, y_train)
y_pred = clf_svc.predict(X_test)

plt.figure(figsize=(10, 6),dpi=100)

plt.grid(True)
a=plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40,cmap='Paired')
b=plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40,cmap='Set1', marker = 'X')

#  Creamos un mesh para evaluar la función de decisión
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
legend1 = ax.legend(*a.legend_elements(), loc="lower left", title="Grupos (Training)")
ax.add_artist(legend1)
legend2 = ax.legend(*b.legend_elements(), loc="upper right", title="Grupos(Test)")
ax.add_artist(legend2)
x = np.linspace(xlim[0], xlim[1], 300)
y = np.linspace(ylim[0], ylim[1], 300)
Y, X = np.meshgrid(y, x)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = clf_svc.decision_function(xy).reshape(X.shape)
# Graficamos el hiperplano y el margen
ax.contour(X, Y,Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['dotted', (5, (10, 3)), 'dotted'])

# Graficamos los vectores soporte
c=ax.scatter(clf_svc.support_vectors_[:, 0], clf_svc.support_vectors_[:, 1], s=10,facecolors='none', edgecolors='k')
Z_pred= clf_svc.predict(np.c_[X.ravel(), Y.ravel()])
Z_pred = Z_pred.reshape(X.shape)

#plt.contourf(XX, YY, Z_pred,cmap = 'spring', alpha = 0.4)
# Printing Accuracy on Training and Test sets

file.write("------------------SVM-----------------------------------------------\n"
f"Training Set Score : {clf_svc.score(X_train, y_train) * 100} %\n"
f"Test Set Score : {clf_svc.score(X_test, y_test) * 100} % \nModel Classification Report : \n{classification_report(y_test, clf_svc.predict(X_test))}")
file.write("--------------------------------------------------------------------")
file.close()
print("------------------SVM-----------------------------------------------")
print(f"Training Set Score : {clf_svc.score(X_train, y_train) * 100} %")
print(f"Test Set Score : {clf_svc.score(X_test, y_test) * 100} %")
 
# Printing classification report of classifier on the test set set data
print(f"\nModel Classification Report : \n{classification_report(y_test, clf_svc.predict(X_test))}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.figure(figsize=(10,8),dpi=100)
sns.set() 
sns.heatmap(conf_matrix.T, cmap='Oranges',annot=True,fmt='d', linewidths=0.5, cbar=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Matriz de confusion de SVM')
print("----------------------------------------------------------------------")

