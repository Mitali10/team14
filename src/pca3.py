# https://stackoverflow.com/questions/65241847/how-to-plot-3d-pca-with-different-colors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from CreditCardData import CreditCardData


# xtrain

obj = CreditCardData()
X_train, X_test, y_train, y_test = obj.split_data()
print("Y COUNT",np.unique(y_train[0], return_counts=True) )
X_resampled, y_resampled = obj.oversample(X_train, y_train)
print("Y RESAMPLED COUNT", np.unique(y_resampled[0], return_counts=True))

sc = StandardScaler()

scaler = StandardScaler()
scaler.fit(X_resampled[0]) 
X_scaled = scaler.transform(X_resampled[0])

pca = PCA(n_components=3)
pca.fit(X_scaled) 
X_pca = pca.transform(X_scaled) 

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)

Xax = X_pca[:,0]
Yax = X_pca[:,1]
Zax = X_pca[:,2]

cdict = {0:'green',1:'red'}
labl = {0:'Autenthic',1:'Fraud'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y_resampled[0]):
 ix=np.where(y_resampled[0]==l)
 ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
           label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("1st Principal Component", fontsize=10)
ax.set_ylabel("2nd Principal Component", fontsize=10)
ax.set_zlabel("3rd Principal Component", fontsize=10)

ax.legend()
plt.show()
