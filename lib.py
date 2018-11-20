#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:04:40 2018

@author: khanhdeux
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def get_iris_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)    
    
    X_combined = np.vstack((X_train, X_test))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))  
    
    return X_train, X_train_std, X_combined, X_combined_std, X_test, y_train, y_combined, y_test   

def get_wine_data():
    df_wine = pd.read_csv('wine.data.txt', header=None)
    df_wine.columns = ['Class label', 
                       'Alcohol',
                       'Malic acid', 
                       'Ash',
                       'Alcalinity of ash', 
                       'Magnesium',
                       'Total phenols', 
                       'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 
                       'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']    
    # print(df_wine.head())
    # print('Class labels', np.unique(df_wine['Class label']))
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values 
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.3, 
                                                        random_state=0, 
                                                        stratify=y)
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    
    return X_train, X_train_norm, X_train_std, X_test, X_test_norm, X_test_std, y_train, y_test

def get_breast_cancer_data():
    df = pd.read_csv('breast_cancer.data.txt', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    print(le.transform(['M','B']))
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=1
                                                        )
    
    return X_train, X_test, y_train, y_test

def get_dummy_data():
    X = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9],
                  [10,11,12]
                  ])
    y = np.array([3,2,1,3])
    return X, y

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution)
    )
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot the class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y==cl, 0],
            y=X[y==cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx,:], y[test_idx]
        
        plt.scatter(
                X_test[:,0], 
                X_test[:,1], 
                c='', 
                edgecolors='black', 
                alpha=1.0,
                linewidths=1, 
                marker='o',
                s=100,
                label='test set')