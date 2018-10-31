# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from lib import plot_decision_regions,get_iris_data
from sklearn.neighbors import KNeighborsClassifier

X_train, X_train_std, X_combined, X_combined_std, X_test, y_train, y_combined, y_test = get_iris_data()
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, 
                      y_combined, 
                      classifier=knn, 
                      test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()