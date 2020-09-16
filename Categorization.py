# -*- coding: utf-8 -*-
"""
@author: Alexander Badrenkov

Machine Learning model used for the automatic categorization of products
"""

#imports
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from collections import OrderedDict

#download just in case
nltk.download('stopwords')
nltk.download('wordnet')

#get training data
train = pd.read_excel('categories_export_v2.xlsx', sheet_name = 'data')
train = train.dropna(subset=['product_type'])

#get test data
test = pd.read_csv('query_result.csv')
test = test.iloc[:, 0:2]
test.columns = ['id', 'product_name']

#exclude a couple hard to determine categories
train = train[train['product_type'] != 'Books']
train = train[train['product_type'] != 'Movies']

#get category counts and remove categories which are too small
train['product_type'] = train['main_category'] + ';' + train['sub_category'] + ';' + train['product_type']
product_type = train['product_type'].value_counts().reset_index()
small_product_type = product_type[product_type['product_type'] < 6]['index']
train = train[~train['product_type'].isin(small_product_type)]

#clean up the product names
train['product_name'] = train['product_name'].str.replace(r"\([^()]*\)","") #text in paranthesis
train['product_name'] = train['product_name'].str.replace('\d+', '') #numbers
train['product_name'] = train['product_name'].str.replace('[^\w\s]','') #punctuation
train['product_name'] = train['product_name'].str.lower() #lowercase
train['product_name'] = train['product_name'].str.replace('[()]', '') #remove parathesis

test['product_name'] = test['product_name'].str.replace(r"\([^()]*\)","") #text in paranthesis
test['product_name'] = test['product_name'].str.replace('\d+', '') #numbers
test['product_name'] = test['product_name'].str.replace('[^\w\s]','') #punctuation
test['product_name'] = test['product_name'].str.lower() #lowercase
test['product_name'] = test['product_name'].str.replace('[()]', '') #remove parathesis

#remove stop words
stop = nltk.corpus.stopwords.words('english')
f = lambda x: ' '.join([item for item in x.split() if item not in stop])
train['product_name'] = train['product_name'].apply(f)

test['product_name'] = test['product_name'].apply(f)

#remove sizing words
banned = ['pack', 'count', 'ct', 'oz', 'pk', 'ea', 'ft', 'floz', 'ltr', 'gal', 'fl', 'ounce', 'size', 'x', 'w', 'xx', 'inch', 'mm', 'lb', 'yds', 'kirkland',
          'instacart', 'costco']
f = lambda x: ' '.join([item for item in x.split() if item not in banned])
train['product_name'] = train['product_name'].apply(f)

test['product_name'] = test['product_name'].apply(f)

#remove duplicate words
train['product_name'] =  (train['product_name'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))

test['product_name'] =  (test['product_name'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))

#lemmatization
lemma = nltk.wordnet.WordNetLemmatizer()
f = lambda x: ' '.join([lemma.lemmatize(item) for item in x.split()])
train['product_name'] = train['product_name'].apply(f)

test['product_name'] = test['product_name'].apply(f)

#get x and y data
x_train = train['product_name']
y_train = train['product_type']
x_test = test['product_name']

#TF-IDF
v = TfidfVectorizer(ngram_range = (1,3))
x_train_idf = v.fit_transform(x_train)
x_test_idf = v.transform(x_test)

#apply model
model = LinearSVC(loss = 'squared_hinge', max_iter = 10000)
model = CalibratedClassifierCV(model, cv = 3)
model.fit(x_train_idf, y_train)

#predict
y_pred = model.predict(x_test_idf)
y_prob = model.predict_proba(x_test_idf)
y_prob = y_prob.max(axis = 1)

#format the output
y_pred = pd.DataFrame(y_pred)
y_pred = pd.DataFrame(y_pred[0].str.split(';').tolist(), columns = ['main_category','sub_category','product_type'])
y_prob = pd.DataFrame(y_prob)
y_prob.columns = ['confidence']
output = pd.concat([test, y_pred, y_prob], axis = 1)

#output
output.to_excel("output.xlsx", index = False)
