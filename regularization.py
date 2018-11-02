# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import get_wine_data
X_train, X_train_norm, X_train_std, X_test, X_test_norm, X_test_std, y_train, y_test = get_wine_data()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Testing accuracy:', lr.score(X_test_std, y_test))
print(lr.intercept_)
print(lr.coef_)

fig = plt.figure()
ax = plt.subplot(111)

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

colors = ['blue', 
          'green', 
          'red', 
          'cyan',
          'magenta', 
          'yellow', 
          'black',
          'pink', 
          'lightgreen', 
          'lightblue',
          'gray', 
          'indigo', 
          'orange']
weights, params = [], []

for c in np.arange(-4., 6.):
    lr = LogisticRegression(
            penalty='l1', 
            C=10.**c,
            random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)    

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1], color=color) 
    
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, 
          fancybox=True)
plt.show()    
    