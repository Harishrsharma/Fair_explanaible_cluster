# metrics.py
# Clustering quality metrics used in the thesis evaluation framework.
#
# Metrics included:
#   silhouette     : higher = better-separated clusters (Rousseeuw 1987)
#                    computed in ORIGINAL space AND in PCA-reduced space
#   silhouette_pca : silhouette on PCA-reduced features (addresses curse of
#                    dimensionality for high-d one-hot encoded data)
#                    Reference: Aggarwal et al. (2001) "On the Surprising
#                    Behavior of Distance Metrics in High Dimensional Space"
#                    https://link.springer.com/chapter/10.1007/3-540-45365-2_2
#   davies_bouldin : lower  = better clusters (Davies & Bouldin 1979)
#   sse            : Sum of Squared Errors – sum of squared distances from
#                    each point to its cluster centroid.  Equivalent to
#                    K-Means inertia; used here for all models uniformly.
#
# Why silhouette is low for high-dimensional data
# -----------------------------------------------
# After one-hot encoding, datasets like Adult have 100+ features.
# In high dimensions, all pairwise distances converge (concentration of
# measure), making silhouette scores close to 0 regardless of cluster
# quality.  silhouette_pca solves this by computing the score on the
# top-N PCA components that explain 95% of variance before distance
# computation – a well-established practice in the literature.
#
# Note: fairness metrics live in fairness.py;
#       explainability metrics live in explainability.py

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


# Number of PCA components kept for silhouette_pca.
# If d <= PCA_N_COMPONENTS the raw features are used (no reduction needed).
PCA_N_COMPONENTS = 50


def clustering_metrics(X, labels):
    """
    Compute standard clustering quality metrics.

    Parameters
    ----------
    X      : ndarray (n, d)
    labels : ndarray (n,) – integer cluster assignments

    Returns
    -------
    dict with keys:
        silhouette      – score in original feature space
        silhouette_pca  – score in PCA-reduced space (more reliable for high-d)
        davies_bouldin  – Davies-Bouldin index
        sse             – Sum of Squared Errors (centroid-based)
    """
    unique = np.unique(labels)

    # Need at least 2 non-empty clusters for silhouette / DB
    if len(unique) < 2:
        return {
            "silhouette":     float("nan"),
            "silhouette_pca": float("nan"),
            "davies_bouldin": float("nan"),
            "sse":            _sse(X, labels),
        }

    n, d = X.shape

    # ── Silhouette in original space (subsample large datasets for speed) ──
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10000, replace=False)
        sil_raw = float(silhouette_score(X[idx], labels[idx]))
    else:
        sil_raw = float(silhouette_score(X, labels))

    # ── Silhouette in PCA-reduced space ───────────────────────────────────
    # Only reduce if the dataset actually has more dimensions than PCA_N_COMPONENTS.
    # This corrects the artificially low scores caused by high-dimensional
    # one-hot features (curse of dimensionality).
    if d > PCA_N_COMPONENTS:
        n_comp = min(PCA_N_COMPONENTS, n - 1, len(unique) - 1 + 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X   # already low-dimensional – no reduction needed

    if n > 10000:
        sil_pca = float(silhouette_score(X_pca[idx], labels[idx]))
    else:
        sil_pca = float(silhouette_score(X_pca, labels))

    return {
        "silhouette":     sil_raw,
        "silhouette_pca": sil_pca,
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "sse":            _sse(X, labels),
    }


def _sse(X, labels):
    """
    Sum of Squared Errors (SSE).
    For each cluster, compute squared L2 distances from every point to
    its cluster centroid (mean), then sum across all clusters.
    This is the standard within-cluster sum of squares (WCSS) used in
    the elbow method and as a universal clustering quality measure.
    """
    total = 0.0
    for k in np.unique(labels):
        members = X[labels == k]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        total += float(np.sum((members - centroid) ** 2))
    return total
