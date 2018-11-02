# -*- coding: utf-8 -*-

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class SBS():
    def __init__(self, 
                 estimator, 
                 k_features, 
                 scoring=accuracy_score, 
                 test_size=0.25, 
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.test_size,
                                                            random_state=self.random_state
                                                            )        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, 
                                 y_train,
                                 X_test,
                                 y_test,
                                 self.indices_
                                 )
        self.scores_ = [score]
        
        print(dim)
        print(self.scores_)
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, 
                                         y_train,
                                         X_test,
                                         y_test,
                                         p
                                         )
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)            
            dim -=1
            self.scores_.append(scores[best])   
            
        self.k_score_ = self.scores_[-1]        
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, 
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
from lib import get_wine_data
X_train, X_train_norm, X_train_std, X_test, X_test_norm, X_test_std, y_train, y_test = get_wine_data()    
    
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)


k_feat = [len(k) for k in sbs.subsets_]
print(k_feat)
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()


# k=3
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

knn.fit(X_train_std, y_train)
print('Traning accuracy:', knn.score(X_train_std, y_train))
print('Testing accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k3], y_train)
print('Traning accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Testing accuracy:', knn.score(X_test_std[:, k3], y_test))
