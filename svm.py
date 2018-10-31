#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:12:11 2018

@author: khanhdeux
"""

import matplotlib.pyplot as plt
from lib import plot_decision_regions,get_iris_data
from sklearn.svm import SVC

X_train, X_train_std, X_combined_std, X_test, y_train, y_combined, y_test = get_iris_data()
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()