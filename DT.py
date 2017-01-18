import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import random
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
import sys
import copy
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import OneHotEncoder
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    print 'Calling learning_curve function'    
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10,         
        n_jobs=1, train_sizes=train_sizes, scoring='accuracy')    
    #print 'train sizes =', train_sizes
    #print 'train scores =', train_scores
    #print 'test scores =', test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    #plt.show()
    #return plt
    return clf, test_scores_mean

def get_list_from_df(df, label_column):
    y = list(df[label_column])
    df.drop([label_column], 1, inplace=True)
    X = df.values.tolist()
    return X, y

filename = 'responses_SMOTE400.csv'
alg = 'dt'
label_column = -1
delim = ','
if len(sys.argv) > 4:
    delim = sys.argv[4]

responsesdf = pd.read_csv(filename, header=None)
newappsdf = pd.read_csv("newApps.csv", header=None)

filedf = pd.concat([responsesdf, newappsdf], ignore_index=True)

le = preprocessing.LabelEncoder()
oh = OneHotEncoder()


newdf = pd.DataFrame()
for i in range(len(filedf.columns)):
    if filedf[i].dtype == object:
        newdf[i] = le.fit_transform(filedf[i])
    else:
        newdf[i] = filedf[i]

filedf = newdf
#print filedf

if (label_column == -1):
    label_column = len(filedf.columns) - 1
#print "label column =", label_column
print
print "This tool will attempt to predict the result of the application"

classifier = ''

if alg == 'dt':
    dt = True
    classifier = 'Decision Trees'
    params = {'max_depth': [5, 10, 15, 20, 25, 30, 40, 50], 
              'criterion': ['gini', 'entropy'],
			  'max_leaf_nodes': [10, 20, 30, 40, 50],
			  'min_weight_fraction_leaf': [0.1, 0.2, 0.3],
                          'random_state': [1]
			  }
    
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=params)


Xall, yall = get_list_from_df(filedf, label_column)

X = np.array(Xall)
y = np.array(yall)
X = X[:len(responsesdf), :]
y = y[:len(responsesdf)]
#X = X[y != 2]
#y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
#print 'lengths of whole data set X and y', len(X), len(y)
#print 'lengths of training set set X and y', len(X_train), len(y_train)
#print 'lengths of test set set X and y', len(X_test), len(y_test)

clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print 
#print "Confusion Matrix for this prediction:"
cnf_matrix = confusion_matrix(y_true, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Yes', 'No'], title='Confusion matrix, without normalization')
plt.savefig("DTConfusionMatrix.png")
#print 'accuracy1 =', acc

loo = LeaveOneOut()
loo.get_n_splits(X)

X = np.array(X)
y = np.array(y)

for train_index, test_index in loo.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train = X[train_index]
    X_test =  X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print (len(X_train), len(X_test), len(y_train), len(y_test))

    '''   
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print 'loo accuracy1 =', acc
    '''


#clf, test_scores_mean = plot_learning_curve(clf, classifier + str(clf.best_params_), np.array(X), np.array(y), 
#                                       #ylim=(0.8, 1.0), #cv =cv,
#                                       train_sizes=np.linspace(.05, 1.0, 5))

#exit()


X_test = np.array(Xall)
y_test = np.array(yall)
X_test = X_test[len(responsesdf):, :]
y_test = y_test[len(responsesdf):]

print 'Cross Validation Accuracy of this Model =', acc
#clf.fit(X, y)
y_true, y_pred = y_test, clf.predict(X_test)

#print "y_true, y_pred =", y_true, y_pred

result = ""
if y_pred == 0:
    result = "Rejected"
else:
    result = "Accepted"

print "This model predicts the applicant will be:", result
#acc = accuracy_score(y_pred, y_test)



'''
print "Best parameters set found on development set:"
print
print clf.best_estimator_
print clf.best_params_
print
print "Grid scores on development set:"
print
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (
        mean_score, scores.std() / 2, params)
print

print "Detailed classification report:"
print
print "The model is trained on the full development set."
print "The scores are computed on the full evaluation set."
print
y_true, y_pred = y_test, clf.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print classification_report(y_true, y_pred)
print 'accuracy last =', acc
print
'''

plt.show()
print
print

