# metrics.py
# Clustering quality metrics used in the thesis evaluation framework.
#
# Metrics included:
#   silhouette     : higher = better-separated clusters (Rousseeuw 1987)
#   davies_bouldin : lower  = better clusters (Davies & Bouldin 1979)
#   inertia        : sum of squared distances to nearest centre (for K-Means)
#
# Note: fairness metrics live in fairness.py;
#       explainability metrics live in explainability.py

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist


def clustering_metrics(X, labels):
    """
    Compute standard clustering quality metrics.

    Parameters
    ----------
    X      : ndarray (n, d)
    labels : ndarray (n,) – integer cluster assignments

    Returns
    -------
    dict with keys: silhouette, davies_bouldin, inertia
    """
    unique = np.unique(labels)

    # Need at least 2 non-empty clusters for silhouette / DB
    if len(unique) < 2:
        return {
            "silhouette":     float("nan"),
            "davies_bouldin": float("nan"),
            "inertia":        _inertia(X, labels),
        }

    # Silhouette on a random subsample if dataset is large (speed)
    n = X.shape[0]
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10000, replace=False)
        sil = silhouette_score(X[idx], labels[idx])
    else:
        sil = silhouette_score(X, labels)

    return {
        "silhouette":     float(sil),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "inertia":        _inertia(X, labels),
    }


def _inertia(X, labels):
    """Sum of squared L2 distances from each point to its cluster centroid."""
    total = 0.0
    for k in np.unique(labels):
        members = X[labels == k]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        total += float(np.sum((members - centroid) ** 2))
    return total
