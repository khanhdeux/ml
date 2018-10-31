#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:12:11 2018

@author: khanhdeux
"""

import matplotlib.pyplot as plt
import numpy as np
from lib import plot_decision_regions,get_iris_data

def gini(p):
    return (p) * (1- (p)) + (1-p) * (1- (1-p))
def entropy(p):
    return - p * np.log2(p) - (1 - p)* np.log2((1-p))
def error(p):
    return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)','Gini Impurity', 'Misclassification Error'], ['-', '-', '--', '-.'],
                          ['black', 'lightgray','red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()    

from sklearn.tree import DecisionTreeClassifier
X_train, X_train_std,X_combined, X_combined_std, X_test, y_train, y_combined, y_test = get_iris_data()

tree = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined, classifier=tree,test_idx=range(105, 150))
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree,
filled=True,
rounded=True,
class_names=['Setosa','Versicolor','Virginica'], 
feature_names=['petal length', 'petal width'], 
out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')