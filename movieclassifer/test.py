# -*- coding: utf-8 -*-

import pickle
import re
import os
from vectorizer import vect

# Unpickle classifiers
clf = pickle.load(open(os.path.join('pkl_objects', 
                                    'classifier.pkl'), 'rb'))

# make prediction for the sentiment analysis
import numpy as np
label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]],
                                               np.max(clf.predict_proba(X))* 100
                                               ))
# Create a new SQLlite database and store 2 examples movie reviews
import sqlite3
import os
if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')
conn = sqlite3.connect('reviews.sqlite')    
c = conn.cursor()
c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", 
          (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", 
          (example2, 0))

conn.commit()
conn.close()

conn= sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()
conn.close()

print(results)