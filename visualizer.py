import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.cm as cmx
from sklearn import tree
from sklearn import metrics
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
#from sknn.mlp import Classifier, Layer
from matplotlib import pyplot as plt
import sys
import copy
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
#import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics import explained_variance_score
from scipy import stats
import scipy as sp
from sklearn import mixture
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.lda import LDA
from sklearn.random_projection import SparseRandomProjection
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.neural_network import MLPClassifier

from time import time

from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def run_clustering(X):
    if clustalg == 'em':
        return run_gmm(X_new)
    if clustalg == 'kmeans':
        return run_kmeans(X_new)

def run_gmm(X):
    #acc = []
    best_acc = 0.0
    best_y_pred = []
    for n_components in range(2, num_clusters+1):
    #for n_components in [num_clusters]:
        for covariance_type in ['spherical', 'tied', 'diag', 'full']:
            clf = mixture.GMM(n_components = n_components, covariance_type = covariance_type)
            clf.fit(X.tolist())
            y_pred = clf.predict(X)
            if len(pd.unique(y_pred.ravel())) == 1:
                continue
            #acc = accuracy_score(y_pred.tolist(), list(y.tolist()))
            acc = silhouette_score(X, y_pred)
            #print 'components =', n_components, 'cov type =', covariance_type, 'accuracy score =', acc
            if acc > best_acc:
                best_acc = acc
                best_n_components = n_components
                best_cov_type = covariance_type
                best_y_pred = y_pred

    print 'best = ', best_acc, best_n_components, best_cov_type
    return best_acc, best_n_components, best_y_pred #, best_cov_type

def run_kmeans(X):
    #print 'run kmeans, X shape =', X.shape
    #acc = 0.0
    best_acc = 0.0
    best_n_clusters = 0
    best_y_pred = []
    #print '------------------------------ num clusters =', num_clusters
    for n in [2]: #range(2, num_clusters+1):
    #for n in [num_clusters]:
        clf = KMeans(n_clusters=n, random_state=3245) #, n_init=50)
        clf.fit(X.tolist())
        y_pred = clf.predict(X)
        #print explained_variance_score(y.tolist(), list(clf.labels_), multioutput='uniform_average')  
        #acc.append(accuracy_score(y.tolist(), list(clf.labels_)))
        #acc = metrics.adjusted_rand_score(y.tolist(), list(clf.labels_))
        #acc = accuracy_score(y.tolist(), list(clf.labels_))
        acc = silhouette_score(X, y_pred)
        #adj_mut.append(metrics.adjusted_mutual_info_score(y.tolist(), list(clf.labels_)))
        #exp_var.append()
        #print "k means with", num_clusters, "clusters"
        #print 'predicted labels labels_ = ', (list(clf.labels_))[:40]
        #print 'actual labels = ', (y.tolist())[:40]
        if acc > best_acc:
            best_acc = acc
            best_n_clusters = n
            best_y_pred = y_pred

    return best_acc, best_n_clusters, best_y_pred

    
        
def get_list_from_df(df, label_column):
    #y = list(df[label_column])
    y = df[label_column]
    df = df.drop([label_column], 1)
    #X = df.values.tolist()
    X = df.values
    return X, y


filename = 'responses_SMOTE400.csv'
clustalg = 'kmeans'
dralg = 'pca'
label_column = -1
delim = ','
if len(sys.argv) > 5:
    delim = sys.argv[5]


responsesdf = pd.read_csv(filename, header=None, delimiter=delim)
newappsdf = pd.read_csv("newApps.csv", header=None)
filedf = pd.concat([responsesdf, newappsdf], ignore_index=True)


#print filedf.ix[len(filedf)-1]
if (label_column == -1):
    label_column = len(filedf.columns) - 1

le = preprocessing.LabelEncoder()

newdf = pd.DataFrame()
for i in range(len(filedf.columns)):
    if filedf[i].dtype == object:
        newdf[i] = le.fit_transform(filedf[i])
    else:
        newdf[i] = filedf[i]

#print newdf
#print(newdf.to_string())
filedf = newdf

#newappsdf = pd.read_csv("newApps.csv", header=None, delimiter=delim)

#print '-------------', len(filedf.columns)
Xall, yall = get_list_from_df(filedf, label_column)
X = np.array(Xall)
y = np.array(yall)
X = X[:len(responsesdf), :]
y = y[:len(responsesdf)]


num_clusters = len(pd.unique(y.ravel()))
#num_clusters = 10
#print 'num clusters = ', num_clusters


if clustalg == 'kmeans':
    best_acc = 0.0
    best_n_clusters = 0
    best_acc, best_n_clusters, y_new = run_kmeans(X)
    #print 'Best kmeans score =', best_acc, 'with', best_n_clusters, 'clusters' 

'''
if clustalg == 'em':
    best_acc = 0.0
    best_n_components = 0
    best_cov_type = ''
    #best_acc, best_n_components, best_cov_type = run_gmm(X)
    best_acc, best_n_components, y_new = run_gmm(X)
    print 'Best kmeans score =', best_acc, 'with', best_n_components, 'clusters' 
'''


'''
params = {'n_clusters': range(1, 31), 
          #'n_init': [50], 
         }

clf = GridSearchCV(KMeans(), param_grid=params)
clf.fit(X.tolist())

#print clf.best_estimator_
#print clf.best_params_
#print
#print 
#print clf.grid_scores_
print "Grid scores on development set:"
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (
        mean_score, scores.std() / 2, params)
print
print clf.scorer_
print
'''


'''
for n in range(1, 29):
    num_clusters = n
    scores = []
    acc, adj_rand, adj_mut = run_kmeans(X, num_clusters)
    print "average k means scores with", num_clusters, "clusters with X unmodified: acc, adj_rand, adj_mut =", acc, adj_rand, adj_mut 

exit()
'''

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig) #, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

pca = PCA()
ica = FastICA()
#logistic = KMeans(n_clusters=num_clusters)

#X, y = get_list_from_df(newdf, label_column)
X = np.array(Xall)
y = np.array(yall) 

print
print "This tool will attempt to show the position of the new applicant wrt to the position of past applicants"


if dralg == 'pca':
    ##################################
    ######## KMeans after PCA ########
    ##################################
    #for n in range(1, len(df.columns) + 1):
    
    for n in [3]: #range(1 ,4):
        pca = PCA(n_components=n)
        X_new = pca.fit_transform(X)
        #print X_new
        #print 'NN accuracy after PCA with n =', n, 'components =', run_nn(X_new, y)
        acc, clusters, y_new = run_kmeans(X_new)
        #print 'NN accuracy after PCA with n =', n, 'components and clustering with', clusters, 'clusters =', run_nn(X_new, y_new)
        #print "average", clustalg, "score after X modified with PCA", n, "components, clusters =", clusters, "silhouette score =", acc
        print "Percentage of data explained by first 3 principal components =", sum(pca.explained_variance_ratio_) * 100
        #print pca.components_
    #print "\n\n\n\n"

#print len(X_new), len(y_new)


'''
#
# 2D plot
#
plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], ['Rejected', 'Accepted']):
    plt.scatter(X_new[y_new == i, 0], X_new[y_new == i, 1], color=color, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()
'''

#
# 3D plot
#
X = X_new
y = np.array(yall)
#X = np.array(Xall)
#y = np.array(yall) 
#print 'type of X, y =', X[-1], y[-1]
y[-1] = 2

for name, label in [('Rejected', 0), ('Accepted', 1), ('New', 2)]:
    #print len(X[y == label])
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), name ,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [0, 1, 2]).astype(np.float)

cm = plt.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=min(y), vmax=max(y))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=scalarMap.to_rgba(y))
scalarMap.set_array(y)
fig.colorbar(scalarMap)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')


#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

xmin = min(X[:, 0])
xmax = max(X[:, 0])
ymin = min(X[:, 1])
ymax = max(X[:, 1])
zmin = min(X[:, 2])
zmax = max(X[:, 2])
ax.set_xlim([xmin-0.5, xmax+0.5])
ax.set_ylim([ymin-0.5, ymax+0.5])
ax.set_zlim([zmin-0.5, zmax+0.5])

#ax.view_init(0, 0)
plt.show()
print "Output saved to newApplicantPosition.png"
plt.savefig("newApplicantPosition.png")
print 
print

'''
#
# Parallel coordinates plot
df = pd.DataFrame(index=None, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Status'])

df['Principal Component 1'] = X[:, 0]/ 10
df['Principal Component 2'] = X[:, 1]/ 10
df['Principal Component 3'] = X[:, 2]/ 10
df['Status'] = y

df.loc[df['Status'] == 0, 'Status'] = 'Rejected'
df.loc[df['Status'] == 2, 'Status'] = 'New'
df.loc[df['Status'] == 1, 'Status'] = 'Accepted'

plt.figure()
from pandas.tools.plotting import parallel_coordinates
filedf.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'status']
print filedf.to_string()
parallel_coordinates(df, 'Status', color=['g', 'b', 'r'])
plt.show()
'''
