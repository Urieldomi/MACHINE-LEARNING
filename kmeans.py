# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:38:59 2022

@author: jorge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#Leemos los datos de entrada para generar el dataset:

X = pd.read_csv('dataset1.csv',header=0)
X = X.loc[:,['h','p']]
X = np.array(X)

clf = KMeans(n_clusters = 5)
clf.fit(X)

centers = clf.cluster_centers_
y = clf.labels_


fig, ax = plt.subplots()
plt.grid(True)
plt.title("Agrupamiento con el algoritmo propio de k-nmeans")
scatter = ax.scatter(X[:,0], X[:,1], s = 20, c = y, cmap = 'plasma', alpha = 0.6)
legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Grupos")
ax.add_artist(legend1)
scatter = ax.scatter(centers[:,0], centers[:,1], s = 20)
plt.show()
