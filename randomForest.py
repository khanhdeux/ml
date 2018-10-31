# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from lib import plot_decision_regions,get_iris_data
from sklearn.ensemble import RandomForestClassifier

X_train, X_train_std, X_combined, X_combined_std, X_test, y_train, y_combined, y_test = get_iris_data()

forest = RandomForestClassifier(criterion='gini', 
                                n_estimators=25, 
                                random_state=1, 
                                n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, 
                      y_combined, 
                      classifier=forest,
                      test_idx=range(105, 150))
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()