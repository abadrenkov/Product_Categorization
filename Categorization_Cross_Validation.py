# -*- coding: utf-8 -*-
"""
@author: Alexander Badrenkov

Cross validation of the machine learning model
"""

#imports
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold

#download in case
nltk.download('stopwords')
nltk.download('wordnet')

#get data
data = pd.read_excel('categories_export_v2.xlsx', sheet_name = 'data')
data = data.dropna(subset=['product_type'])

#exclude a couple hard to determine categories
data = data[data['product_type'] != 'Books']
data = data[data['product_type'] != 'Movies']

#get category counts
data['product_type'] = data['main_category'] + ';' + data['sub_category'] + ';' + data['product_type']
product_type = data['product_type'].value_counts().reset_index()
small_product_type = product_type[product_type['product_type'] < 6]['index']
data = data[~data['product_type'].isin(small_product_type)]

#clean up the product names
data['product_name'] = data['product_name'].str.replace(r"\([^()]*\)","") #text in paranthesis
data['product_name'] = data['product_name'].str.replace('\d+', '') #numbers
data['product_name'] = data['product_name'].str.replace('[^\w\s]','') #punctuation
data['product_name'] = data['product_name'].str.lower() #lowercase
data['product_name'] = data['product_name'].str.replace('[()]', '') #remove parathesis

#remove stop words
stop = nltk.corpus.stopwords.words('english')
f = lambda x: ' '.join([item for item in x.split() if item not in stop])
data['product_name'] = data['product_name'].apply(f)

#remove sizing words
banned = ['pack', 'count', 'ct', 'oz', 'pk', 'ea', 'ft', 'floz', 'ltr', 'gal', 'fl', 'ounce', 'size', 'x', 'w', 'xx', 'inch', 'mm', 'lb', 'yds', 'kirkland',
          'instacart', 'costco']
f = lambda x: ' '.join([item for item in x.split() if item not in banned])
data['product_name'] = data['product_name'].apply(f)

#remove duplicate words
data['product_name'] =  (data['product_name'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))

#lemmatization
lemma = nltk.wordnet.WordNetLemmatizer()
f = lambda x: ' '.join([lemma.lemmatize(item) for item in x.split()])
data['product_name'] = data['product_name'].apply(f)

miss_test = pd.DataFrame({'id' : [], 'product_name' : [], 'product_type' : []})
miss_pred = pd.DataFrame({'prediction' : []})
miss_prob = pd.DataFrame({'confidence' : []})

#stratified kfold split
skf = StratifiedKFold(n_splits = 5, random_state = 3)
for train_index, test_index in skf.split(data['product_name'], data['product_type']):
    train = data.iloc[train_index, :]
    test = data.iloc[test_index, :]
    x_train = train['product_name']
    y_train = train['product_type']
    x_test = test['product_name']
    y_test = test['product_type']

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
    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)
    
    temp_test = test.where(y_test != y_pred)
    temp_test = temp_test[['id', 'product_name', 'product_type']]
    temp_test = temp_test.dropna().reset_index(drop=True)
    miss_test = miss_test.append(temp_test)
    
    temp_pred = pd.DataFrame(y_pred[np.where(y_test != y_pred)]).reset_index(drop=True)
    temp_pred.columns = ['prediction']
    miss_pred = miss_pred.append(temp_pred)
    
    y_prob = y_prob.max(axis = 1)
    temp_prob = pd.DataFrame(y_prob[np.where(y_test != y_pred)]).reset_index(drop=True)
    temp_prob.columns = ['confidence']
    miss_prob = miss_prob.append(temp_prob)

#print mistakes
truth = pd.DataFrame(miss_test['product_type'].str.split(';').tolist(), columns = ['neg_main_category','neg_sub_category','neg_product_type']).reset_index(drop=True)
prediction = pd.DataFrame(miss_pred['prediction'].str.split(';').tolist(), columns = ['model_main_category','model_sub_category','model_product_type']).reset_index(drop=True)
id_col = miss_test[['id', 'product_name']].reset_index(drop=True)
miss_prob = miss_prob.reset_index(drop=True)
miss = pd.concat([id_col, truth, prediction, miss_prob], axis = 1)
miss = miss.sort_values('confidence', ascending = False)
miss.to_excel("outputMistakes.xlsx", index = False)
