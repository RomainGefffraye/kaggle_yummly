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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("train.json")
traindf['ingredients_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

testdf = pd.read_json("test.json")
testdf['ingredients_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]

corpustr = traindf['ingredients_string']

vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", binary=False , token_pattern=r'\w+' , sublinear_tf=False)

predictor_tr = vectorizertr.fit_transform(corpustr).todense()

# best for now : C = 0.46, penalty = "l2"  = 0.788580447312


targets_tr = traindf['cuisine']

#clf = ExtraTreesClassifier(n_estimators=100, max_depth=None min_samples_split=1, random_state=0, verbose= 2,n_jobs = -1)
penalty = ["l1"]#, "l2"]

for j in penalty:
    for i in range(1,100):
        p = i*0.1
        print p
        clf = LinearSVC(C=p, penalty=j, dual=False)
#clf = GaussianNB()
#clf = svm.SVC(verbose= 2)
        clf = clf.fit(predictor_tr, targets_tr)
        scores = cross_val_score(clf, predictor_tr, targets_tr)
        print scores.mean()
'''
corpusts = testdf['ingredients_string']
tfidfts=vectorizertr.transform(corpusts).todense()

prediction = clf.predict(tfidfts)
testdf['cuisine']= prediction
testdf = testdf.sort('id' , ascending=True)
testdf[['id', 'cuisine' ]].to_csv("submission.csv")
'''
