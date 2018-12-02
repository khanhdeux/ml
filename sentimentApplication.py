# -*- coding: utf-8 -*-

import pyprind
import pandas as pd
import os

# change the basepath directory
# to the unzipped movie dataset

#basepath = 'aclImdb'
#labels = {'pos' : 1, 'neg' : 0}
#pbar = pyprind.ProgBar(50000)
#df = pd.DataFrame()
#for s in ('test', 'train'):
#    for l in ('pos', 'neg'):
#        path = os.path.join(basepath, s, l)
#        for file in os.listdir(path):
#            with open(os.path.join(path, file),
#                      'r', encoding='utf-8') as infile:
#                txt = infile.read()
#            df = df.append([[txt, labels[l]]], ignore_index=True)    
#            pbar.update()
#
# df.columns = ['review', 'sentiment']
#
#import numpy as np
#np.random.seed(0)
#df = df.reindex(np.random.permutation(df.index))
#df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# END RENDERING RAW SOURCE INTO movie_data.csv

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

# Using Countvectorizer to transform text into feature vectors counts of 
# how often the words occur in particular document

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
           'The sun is shining',
           'The weather is sweet',
           'The sun is shining, the weather is sweet, and one and one is two'
       ])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

# Using Td-idf to downweight the occurencies in feature vectors
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True
                         )
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

print(df.loc[0, 'review'][-50:])

# Using regular expression for filtering special characters
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


# Checks if the preprocessor works correctly
print(preprocessor(df.loc[0, 'review'][-50:]))
df['review'] = df['review'].apply(preprocessor)

# Split the text into individual elements
def tokenizer(text):
    return text.split()

print(tokenizer('runners like running and thus they run'))

# Wordstemming for transforming a word into its root form
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

# Using stop-word removal to remove words have no useful information
# that can distinguish between different classes of documents
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
         if w not in stop  
       ])
    
# Devide the DataFrame into 25k traning data and 25k testing data
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Use grid search with 5-fold cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train[:10], y_train[:10])
print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Stochastic gradient descent

# Tockenizer cleans the unprocessed data in movie_data.csv
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
print(tokenizer('a runner likes running and runs a lot'))

# stream_docs that reads and return one document at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
print(next(stream_docs(path='movie_data.csv')))

# stream_docs that reads a number of documents specified by size parameters
def get_minibatch(doc_stream, size):
    docs, y =[], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y        

# Hashvectorize - data independent
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer
                         )    
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

# Start out of core learning
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
    
# Use the last 5000 samples for evaluating the performance
X_test, y_test = get_minibatch(doc_stream, size=5000)    
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# Training the test data
clf = clf.partial_fit(X_test, y_test)    


# Laten dirichlet allocation for topic modeling - clustering task
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000
                        )
X = count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch'
                                )
X_topics = lda.fit_transform(X)
print(lda.components_.shape)

# Print the most important words for each of the 10 topics
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic: %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    
# Plot 3 movies from horror category (category 6, index position 5)    
horror = X_topics[:, 5].argsort()[::-1]    
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')          
