# https://stackoverflow.com/questions/65241847/how-to-plot-3d-pca-with-different-colors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CreditCardData import CreditCardData

ccd = CreditCardData()
X = ccd.data
y = ccd.labels

def plot(X, y):
    sc = StandardScaler()
    scaler = StandardScaler()
    scaler.fit(X) 
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=3)
    pca.fit(X_scaled) 
    X_pca = pca.transform(X_scaled) 

    ex_variance=np.var(X_pca,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)

    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
    Zax = X_pca[:,2]

    cdict = {0:'red',1:'green'}
    labl = {0:'Autenthic',1:'Fraud'}
    marker = {0:'*',1:'o'}
    alpha = {0:.3, 1:.5}

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix=np.where(y==l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
        label=labl[l], marker=marker[l], alpha=alpha[l])
    # for loop ends
    ax.set_xlabel("1st Principal Component", fontsize=10)
    ax.set_ylabel("2nd Principal Component", fontsize=10)
    ax.set_zlabel("3rd Principal Component", fontsize=10)

    ax.legend()
    plt.show()
scaler = StandardScaler()
scaler.fit(X) 
X_scaled = scaler.transform(X)

pca = PCA(n_components=2)
pca.fit(X_scaled) 
X_pca = pca.transform(X_scaled) 

Xax = X_pca[:,0]
Yax = X_pca[:,1]
# Zax = X_pca[:,2]

cdict = {0:'green',1:'red'}
labl = {0:'Autenthic',1:'Fraud'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

fig.patch.set_facecolor('white')
for l in np.unique(y):
 ix=np.where(y==l)
 ax.scatter(Xax[ix], Yax[ix],  c=cdict[l], s=40,
           label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("1st Principal Component", fontsize=10)
ax.set_ylabel("2nd Principal Component", fontsize=10)
# ax.set_zlabel("3rd Principal Component", fontsize=10)

ax.legend()
plt.savefig("credit_card_data_2d.png", dpi=600)
plt.show()
