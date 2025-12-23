import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def micro_macro_f1(y_true, y_pred):
    labels = np.unique(y_true)
    
    TP = FP = FN = 0
    f1_per_class = []

    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        TP += tp
        FP += fp
        FN += fn

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)

        f1_per_class.append(f1)

    micro_p = TP / (TP + FP)
    micro_r = TP / (TP + FN)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    macro_f1 = np.mean(f1_per_class)

    return micro_f1, macro_f1


def kmeans_cluster(X, k):
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(X)
    return labels

def hierarchical_cluster(X, k):
    model = AgglomerativeClustering(n_clusters=k, linkage='average')
    labels = model.fit_predict(X)
    return labels

def dbscan_cluster(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

def silhouette(X, labels):
    return silhouette_score(X, labels)

def pca_reduce(X, n_components=1):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def visualize_scores_1d(X_1d, labels, title):
    plt.figure()
    plt.scatter(X_1d[:, 0], np.zeros_like(X_1d[:, 0]), c=labels)
    plt.title(title)
    plt.yticks([])
    plt.show()
def visualize_scores_2d(X, labels, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("Course 1")
    plt.ylabel("Course 2")
    plt.title(title)
    plt.show()
