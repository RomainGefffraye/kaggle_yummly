from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
from sklearn.ensemble import ExtraTreesClassifier
# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("train.json")
traindf['ingredients_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

testdf = pd.read_json("test.json")
testdf['ingredients_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]

corpustr = traindf['ingredients_string']

vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", binary=False , token_pattern=r'\w+' , sublinear_tf=False)

predictor_tr = vectorizertr.fit_transform(corpustr).todense()

targets_tr = traindf['cuisine']

clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
   min_samples_split=1, random_state=0, verbose= 2)
clf = clf.fit(predictor_tr, targets_tr)

corpusts = testdf['ingredients_string']
tfidfts=vectorizertr.transform(corpusts)

prediction = clf.predict(tfidfts)
testdf['cuisine']= prediction
testdf = testdf.sort('id' , ascending=True)
testdf[['id', 'cuisine' ]].to_csv("submission.csv")
