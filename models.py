from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import numpy as np

def do_pca(X, n=2):
    pca = PCA(n_components=n)
    return pca.fit_transform(X), pca

def do_kernel_pca(X, n=2, kernel='rbf'):
    kpca = KernelPCA(n_components=n, kernel=kernel)
    return kpca.fit_transform(X), kpca

def do_lda(X, y, n=2):
    lda = LinearDiscriminantAnalysis(n_components=n)
    return lda.fit_transform(X, y), lda

def do_svd(X, n=2):
    svd = TruncatedSVD(n_components=n)
    return svd.fit_transform(X), svd

def naive_bayes(X, y):
    m = GaussianNB()
    m.fit(X, y)
    return m

def decision_tree(X, y, depth=None):
    m = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
    m.fit(X, y)
    return m

def knn(X, y, k=1, metric='euclidean'):
    m = KNeighborsClassifier(n_neighbors=1, weights='distance', metric=metric)
    m.fit(X, y)
    return m

def lda_classifier(X, y):
    m = LinearDiscriminantAnalysis()
    m.fit(X, y)
    return m

def logistic_reg(X, y):
    m = LogisticRegression(max_iter=5000, C=1000.0)
    m.fit(X, y)
    return m

def neural_net_clf(X, y, layers=(200, 100, 50)):
    m = MLPClassifier(hidden_layer_sizes=layers, max_iter=5000, alpha=0.0, random_state=42)
    m.fit(X, y)
    return m

def linear_reg(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m

def neural_net_reg(X, y, layers=(64, 32)):
    m = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000)
    m.fit(X, y)
    return m

from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleBBN(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = GaussianNB()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def bayesian_network(X, y):
    m = SimpleBBN()
    m.fit(X, y)
    return m

def split_data(X, y, test_size=0.2, stratify=None):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def get_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    return train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)
