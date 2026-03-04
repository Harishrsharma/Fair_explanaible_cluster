# clustering.py

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def run_kmeans(X, k, random_state):
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X)
    return labels


def kmedoids(X, k, max_iter=100, random_state=42):

    np.random.seed(random_state)
    n = X.shape[0]
    medoids = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):

        distances = cdist(X, X[medoids])
        labels = np.argmin(distances, axis=1)

        new_medoids = []

        for i in range(k):
            cluster_points = np.where(labels == i)[0]

            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue

            D = cdist(X[cluster_points], X[cluster_points])
            total = D.sum(axis=1)
            best = cluster_points[np.argmin(total)]
            new_medoids.append(best)

        new_medoids = np.array(new_medoids)

        if np.all(new_medoids == medoids):
            break

        medoids = new_medoids

    return labels
