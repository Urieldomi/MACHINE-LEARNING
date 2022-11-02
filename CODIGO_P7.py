# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:13:03 2022

@author: jorge
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score


"""
Cargando dataset
"""

"Dataset 1"
data_1 = pd.read_csv('dataset_classifiers1.csv')
D1train_data = data_1.loc[:,['0','1']]
D1train_data = np.array(D1train_data)

D1train_target = data_1.loc[:,['y_true']]
D1train_target = np.array(D1train_target)


D1 = [(D1train_data),(D1train_target)]

"Dataset 2"
data_2 =pd.read_csv('dataset_classifiers2.csv')

D2train_data = data_2.loc[:,['0','1']]
D2train_data = np.array(D2train_data)

D2train_target = data_2.loc[:,['y_true']]
D2train_target = np.array(D2train_target)

D2 =[(D2train_data),(D2train_target)]

"Dataset 3"

data_3 =pd.read_csv('dataset_classifiers3.csv')

D3train_data = data_3.loc[:,['0','1']]
D3train_data = np.array(D3train_data)

D3train_target = data_3.loc[:,['y_true']]
D3train_target = np.array(D3train_target)

D3 =[(D3train_data),(D3train_target)]



names = [
    "Nearest Neighbors",
    "RBF SVM",
    "Perceptron"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(C=10,gamma=0.1),
    Perceptron(alpha=0.3,max_iter=5)
]


datasets = [
    D1,
    D2,
    D3
    
]
file = open("Reporte de Clasificacion.txt","w")
figure = plt.figure(figsize=(12, 7),dpi=200)
i = 1

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data",size =3)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.6
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1



    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap='bwr', alpha=0.6, ax=ax, eps=0.5
        )
        
        #Validacion cruzada
        
        scores= cross_val_score(clf, X, y, cv=10)
        file.write( f"Dataset :{ds_cnt+1} " f"Clasificador {name}"
              f"\n"
              f"Acuraccy para cada uno de los 10 folds :{scores}"
              f"\n\n")
        
        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr',alpha= 0.1
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap='bwr',
             s = 1,
            alpha=0.1,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name,size =3)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=2,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
file.close()

