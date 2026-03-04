# metrics.py

from sklearn.metrics import silhouette_score, davies_bouldin_score


def clustering_metrics(X, labels):

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels)
    }
