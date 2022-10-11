 
"""
@author: jorge uriel
KNN_ALGORTIMO_MACHINE LEARNING 2022

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statistics import mode


#*******************************************************************************
#creamos una  la función que calcula la distancia al cuadrado en este caso es la 
#ecludiana para vectores de n dimensiones
def dist_eu(x,y):
  
  distancia = 0 #INICIAMOS CON 0 EN LA DISTANCIA
  n = len(x) #LA LONGITUD DEL VECTOR X
  
  for i in range(n):
      distancia += (x[i] - y[i])**2 #CALCULANDO CON LA FORMA DEL DISTANCIA
  return distancia

#Se define la función kNN
def kNN_alg(k, X_train, y_train, X_test):
    n = len(X_train) #CALCULANDO LONGITUD EL VECTOR X_TRAIN
    m = len(X_test)  #CALCULANDO LONGITUD EL VECTOR X_Test
    k_vecinos = np.zeros(k)               
    k_vecinos_class = np.zeros(k)         
                                            
    y_test = np.zeros(m)
    
    #Se Calcula las distancias los puntos de z a cada uno de los puntos del conjunto X
    for l in range (m):                    
    
        distancias = np.zeros(n)        
        
        for j in range (n):               
            distancias[j] = dist_eu(X_train[j], X_test[l]) 
        #Se calculan los k puntos más cercanos al l-ésimo elemento de X_test
        for i in range (k):
            n_vecino = np.argmin(distancias)
            k_vecinos[i] = n_vecino
            k_vecinos_class[i] = y_train[n_vecino]
            np.delete(distancias, n_vecino)
            
            
            y_test[l] = mode(k_vecinos_class)
            
    y_test = [int(i) for i in y_test]

    y_test = np.array(y_test)

            
    return y_test


#******************************************************************************
#Se genera un dataset
data_coords, data_labels = make_blobs(n_samples = 250, centers = 3, cluster_std = 1.0, random_state = 4)

#Se hace unas particiones 
training_data, test_data, training_labels, test_labels = train_test_split(data_coords, data_labels, test_size = 0.2, shuffle = True, random_state = 4)

#Se calcula con algoritmo kNN_alg
y_kNN_alg = kNN_alg(5, training_data, training_labels, test_data)

#Se calcula con  KNeighborsClassifier de sklearn
knn_sklearn= KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(training_data, training_labels)
y_kNN_Sklearn = knn_sklearn.predict(test_data)

#Se grafica el conjunto de entrenamiento y el de prueba, por el algoritmo de k-NN_alg
fig, ax = plt.subplots()
plt.title("Algoritmo k-NN_alg vs Algortimo de sklearn de Knn")
scatter = ax.scatter(training_data[:,0], training_data[:,1], s = 20, alpha=0.70,c = training_labels, cmap = 'tab10')
leyendas_group1 = ax.legend(*scatter.legend_elements(),fontsize='x-small',borderpad=0.2, loc="lower right", title="Cluster(Training)")
ax.add_artist(leyendas_group1)
scatter = ax.scatter(test_data[:,0], test_data[:,1], s = 30,alpha=0.70,c = y_kNN_Sklearn, cmap = 'viridis')
scatter2 = ax.scatter(test_data[:,0], test_data[:,1], s = 5,marker='v' ,c = y_kNN_alg, cmap = 'Paired')
leyendas = ax.legend(*scatter.legend_elements(), fontsize='x-small',borderpad=0.2,title_fontsize='x-small',loc="upper left",title="Cluster_sklearn(Test)")
leyendas2 = ax.legend(*scatter2.legend_elements(), fontsize='x-small',borderpad=0.2,title_fontsize='x-small',loc="lower center", title="Cluster_k_NN(Test)")
ax.add_artist(leyendas)
plt.show()


#Se grafica el conjunto de entrenamiento y el de prueba, clasificado por el algoritmo de sklearn de k-nn


print(f"\nEn el algoritmo de sklearn se tiene una precisión de {knn_sklearn.score(test_data, test_labels)*100}%, con este  reporte de clasificación tenemos :\n\n{classification_report(test_labels, y_kNN_Sklearn)}")
print(f"\nEn el algoritmo k-NN_alg se tiene una precisión de {accuracy_score(test_labels, y_kNN_alg)*100}%, con este reporte de clasificación tenemos : \n\n{classification_report(test_labels, y_kNN_alg)}")
