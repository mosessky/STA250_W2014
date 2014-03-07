from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl


#simple data processing funciton 
import recProcess

#different machine learning combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
categories = ['not a real question', 'not constructive', 'off topic', 'open', 'too locailized']

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
op.add_option("--train_filename",
              action="store", type = str, dest="trainfilename", default="train.csv",
              help ="Input file absolute directory.")
op.add_option("--test_filename",
              action="store", type = str, dest="testfilename", default="train-sample.csv")
#op.add_option("--output_filename", action="store_true", 
   #           type = str, dest="outputfilename", default="submission.csv")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

print("load training file")
data_train =  recProcess.processdata(opts.trainfilename)

print("load test file")
data_test = recProcess.processdata(opts.testfilename)

print("%d documents  (training set)" % (
    len(data_train[0])))
print("%d documents  (test set)" % (
    len(data_test[0])))
print('data loaded')
print()

y_train, y_test = data_train[0], data_test[0]
x_train, x_test = data_train[1], data_test[1]



t0 = time()
print("Extracting features from the train dataset")

if opts.use_hashing:
   vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   ngram_range = (1,2))
   x_train = vectorizer.transform(x_train)
else:    
   vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
   x_train = vectorizer.fit_transform(x_train)


duration = time() - t0
print("done in %fs" % duration)
print("training data shape: n_samples: %d, n_features: %d" % x_train.shape)
print()

t0 = time()
print("Extracting features from the test dataset")
x_test = vectorizer.transform(x_test)
duration = time() - t0
print("done in %fs " % duration)
print("testing data shape: n_samples: %d, n_features: %d" % x_test.shape)
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    print("done in %fs" % (time() - t0))
    print()
    
def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."

if opts.use_hashing:
    feature_names = None
else:
    features_names = np.asarray(vectorizer.get_feature_names())

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    #we report f_score here, which is a balance parameter between accuracy and recall
    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:       
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

#test different machine learning methods on the training and testing data
results = []
for clf, name in (
       # (RidgeClassifier(tol=1e-1, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
       # (KNeighborsClassifier(n_neighbors=8), "kNN")):
    
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))

def getscore(filename = "/home/shikai/Downloads/vowpal_wabbit-7.2/vowpalwabbit/data/precise_predictions.txt"):
    f1 = open(filename, "r")
    rst = []
    for i in range(len(y_test)):
        line = f1.readline()
        try:
            strs = line.split()
            rst.append(categories[int(float(strs[0])) -1])
        except:
            continue
    #we report f_score here, which is a balance parameter between accuracy and recall
    score = metrics.f1_score(y_test, rst)
    return score       
    

# make some plots
#indices = np.arange(len(results))
results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
#I add two other methods that I find from website
#They rank #10 and #11 in public board
clf_names.append("zzgmnuntz")
scorezz = getscore()
score.append(scorezz)
training_time.append(10)
test_time.append(0.1)
clf_names.append("Lui")
scorelui=getscore("/home/shikai/Downloads/vowpal_wabbit-7.2/vowpalwabbit/data/en_precise_predictions.txt")
score.append(scorelui)
training_time.append(10)
test_time.append(0.1)

indices = np.arange(len(score))

training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.savefig('compare_alg.png')
pl.show()


