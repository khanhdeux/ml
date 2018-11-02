# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.isnull().sum())
print(df.values)
print(df.dropna(axis=0))
print(df.dropna(axis=1))

# drop when all collums or rows are NAN
print('-----')
print(df.dropna(how='all'))

# drop rows that have less than 4 real values
print('-----')
print(df.dropna(thresh=4))

# only drop rows where NAN appear in specific columns
print('-----')
print(df.dropna(subset=['C']))


# mean imputation
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print('-----')
print(df.values)
print(imputed_data)

df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1'],        
        ])
df.columns = ['color', 'size', 'price', 'classlabel']
print('-----')
print(df)

size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1
        }
df['size'] = df['size'].map(size_mapping)
print(df)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(inv_size_mapping)
print(df['size'].map(inv_size_mapping))
print(df)
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print('-----')
print(y)
print(class_le.inverse_transform(y))

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0]= color_le.fit_transform(X[:, 0])
print('-----')
print(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0], sparse=False)
print('-----')
print(X)
print(ohe.fit_transform(X))

print('-----')
print(pd.get_dummies(df[['price', 'color', 'size']]))
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

print('-----')
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray()[:, 1:])

ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean() / ex.std()))
print('normilized:', (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.ensemble import RandomForestClassifier
from lib import get_wine_data
import matplotlib.pyplot as plt

X_train, X_train_norm, X_train_std, X_test, X_test_norm, X_test_std, y_train, y_test = get_wine_data()

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

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print ("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center'
        )
plt.xticks(
        range(X_train.shape[1]), 
        feat_labels[indices],
        rotation=90
        )
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of samples that meet this criterion:', X_selected.shape[0])
for f in range(X_selected.shape[1]):
     print ("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    

