from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
iris = load_iris()
clf = tree.DecisionTreeClassifier()
X = [[0,0],[1,1],[1,0],[0,1]]
Y = [0,1,0,0]
clf = clf.fit(X, Y)


dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,feature_names=['A','B'],
                        class_names=['False','True'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")



D = ['I hate Databases', 'I like Databases']
vectorizertr = TfidfVectorizer(
                             ngram_range = ( 1 , 2 ),analyzer="word", binary=False , token_pattern=r'\w+' , sublinear_tf=False)

tfidftr=vectorizertr.fit_transform(D).todense()
#print tfidftr
