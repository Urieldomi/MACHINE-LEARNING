# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:41:35 2022

@author: jorge
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
X = pd.read_csv('dataset2.csv')
X = pd.DataFrame.to_numpy(X)

pca = PCA(n_components = 2)
pca.fit(X)

X_reduced = pca.transform(X)
plt.figure(figsize=(10,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], s=50, cmap = 'winter')
plt.grid(True)
plt.show()

clf = KMeans(n_clusters = 5)
clf.fit(X_reduced)

centers = clf.cluster_centers_
y = clf.labels_

fig, ax = plt.subplots(figsize=(10,6))
plt.grid(True)
plt.title("Agrupamiento con el algoritmo propio de k-nmeans")
scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], s = 20, c = y, cmap = 'plasma', alpha = 0.6)
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Grupos")
ax.add_artist(legend1)
scatter = ax.scatter(centers[:,0], centers[:,1], s = 20)
plt.show()