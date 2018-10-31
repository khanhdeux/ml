#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:13:31 2018

@author: khanhdeux

for each iter
    update = eta * Sum [i 0 -> X.shape[0]]((y[i]) - net_input(x[i]))* x[i])
    w[j] += update

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import plot_decision_regions

df = pd.read_csv('iris.data.txt', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

class AdalineGD(object):
    def __init__(self, eta = 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter=n_iter
        self.random_state = random_state        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])  
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            errors = (y - self.activation(net_input))
            self.w_[1:] += self.eta * np.dot(X.T, errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/ 2.0
            self.cost_.append(cost)
        return self    
    def activation(self,X):
        return X
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >=0.0, 1, -1)   

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum square error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum square error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(eta=0.01, n_iter=15)
ada.fit(X_std, y)

plot_decision_regions(X_std,y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [stadardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-square-error')
plt.show()