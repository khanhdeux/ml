#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:13:31 2018

@author: khanhdeux

for each xi 
    update = eta * (y[i]) - net_input(x[i]))* x[i]
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

class AdalineSGD(object):
    def __init__(self, eta = 0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter=n_iter
        self.random_state = random_state        
        self.shuffle = shuffle
        self.w_initialized = False
    def fit(self,X,y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    def _update_weights(self, xi, target):
        net_input = self.net_input(xi)
        error = (target - self.activation(net_input))
        cost = 0.5 * (error**2)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        return cost        
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)                
        else:
            self._update_weights(X,y)
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
        self.w_initialized = True
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    def activation(self,X):
        return X
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >=0.0, 1, -1)   

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
ada.fit(X_std[0:65,:], y[0:65])
ada.partial_fit(X_std[65:100,:], y[65:100])

plot_decision_regions(X_std,y, classifier=ada)
plt.title('Adaline - Stochastis Gradient Descent')
plt.xlabel('sepal length [stadardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')
plt.show()