# clustering.py
# Clustering algorithms used in the thesis pipeline:
#   1. K-Means   (vanilla baseline, sklearn)
#   2. K-Medoids (PAM-style baseline)
#   3. K-Medians (professor-requested baseline)
#
# Reference for K-Medoids / K-Medians:
#   Kaufman & Rousseeuw (1987) "Clustering by Means of Medoids"

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# K-Means (sklearn wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X, k, random_state=42):
    """Standard K-Means clustering. Returns integer label array."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# K-Medoids (PAM – Partitioning Around Medoids)
# ─────────────────────────────────────────────────────────────────────────────

def kmedoids(X, k, max_iter=100, random_state=42):
    """
    K-Medoids clustering.
    Medoids are actual data points that minimise total intra-cluster distance.
    More robust to outliers than K-Means; produces exemplar-based explanations.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    medoid_idx = rng.choice(n, k, replace=False)

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        dists = cdist(X, X[medoid_idx])          # (n, k)
        labels = np.argmin(dists, axis=1)

        # Update medoids: pick point with min total distance to cluster members
        new_medoids = np.empty(k, dtype=int)
        for i in range(k):
            members = np.where(labels == i)[0]
            if len(members) == 0:
                new_medoids[i] = medoid_idx[i]   # keep current if empty
                continue
            D = cdist(X[members], X[members])
            new_medoids[i] = members[np.argmin(D.sum(axis=1))]

        if np.all(new_medoids == medoid_idx):
            break
        medoid_idx = new_medoids

    # Final assignment
    dists = cdist(X, X[medoid_idx])
    labels = np.argmin(dists, axis=1)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# K-Medians
# ─────────────────────────────────────────────────────────────────────────────

def k_medians(X, k, max_iter=100, random_state=42):
    """
    K-Medians clustering.
    Cluster centres are computed as component-wise medians (L1 minimiser).
    More robust to outliers than K-Means; uses L1 (Manhattan) distance for
    assignment to stay consistent with the median centre.

    Added as explicit baseline per professor's request.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Initialise centres from random data points
    init_idx = rng.choice(n, k, replace=False)
    centers = X[init_idx].astype(float)

    for _ in range(max_iter):
        # Assign using L1 distance (consistent with median centres)
        dists = cdist(X, centers, metric="cityblock")   # (n, k)
        labels = np.argmin(dists, axis=1)

        # Update centres as component-wise medians
        new_centers = np.empty_like(centers)
        for i in range(k):
            members = X[labels == i]
            if len(members) == 0:
                new_centers[i] = centers[i]
            else:
                new_centers[i] = np.median(members, axis=0)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Final assignment
    dists = cdist(X, centers, metric="cityblock")
    labels = np.argmin(dists, axis=1)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Elbow method: automatic k selection
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X, k_range=(2, 11), random_state=42):
    """
    Select optimal number of clusters via the elbow method on KMeans inertia.

    Algorithm
    ---------
    1. Fit KMeans for each k in k_range on a subsample (≤5 000 points) for speed.
    2. Compute the normalised second derivative of the inertia curve.
    3. Return the k at the sharpest elbow (maximum second-derivative, i.e.,
       where the rate of improvement slows most abruptly).

    Normalisation before second-derivative ensures the elbow detection is
    scale-independent (works whether inertia is 1e3 or 1e8).

    Parameters
    ----------
    X         : ndarray (n, d)
    k_range   : tuple (k_min, k_max_exclusive)  – range of k values to try
    random_state : int

    Returns
    -------
    optimal_k : int
    inertias  : list of float – inertia for each k tried (useful for plotting)
    ks        : list of int   – k values tried
    """
    ks = list(range(k_range[0], k_range[1]))

    # Subsample large datasets so elbow search stays fast
    X_fit = X
    if X.shape[0] > 5000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], 5000, replace=False)
        X_fit = X[idx]

    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        km.fit(X_fit)
        inertias.append(km.inertia_)

    inertias_arr = np.array(inertias, dtype=float)

    if len(ks) < 3:
        return ks[0], inertias, ks   # not enough points to find elbow

    # Normalise inertia to [0, 1] so scale doesn't affect elbow detection
    span = inertias_arr[0] - inertias_arr[-1]
    if span < 1e-10:
        return ks[0], inertias, ks   # flat curve – default to smallest k
    inertias_norm = (inertias_arr - inertias_arr[-1]) / span

    # Second derivative: where the normalised curve bends most sharply
    d1 = np.diff(inertias_norm)   # rate of decrease per step
    d2 = np.diff(d1)              # rate-of-change of that rate

    # argmax(d2): the index (in d2) where decrease slows most abruptly
    # d2 has len(ks)-2 elements; d2[i] corresponds to ks[i+2]
    elbow_idx = int(np.argmax(d2)) + 2
    elbow_idx = min(elbow_idx, len(ks) - 1)   # safety clamp

    # Never return fewer than 3 clusters (too coarse for thesis datasets)
    optimal_k = max(3, ks[elbow_idx])

    return optimal_k, inertias, ks


# ─────────────────────────────────────────────────────────────────────────────
# Cluster on fairlet centres (used after fairlet decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_fairlet_centers(fairlet_centers, k, method="kmedoids", random_state=42):
    """
    Run clustering on the reduced set of fairlet medoid points.
    Returns label array over fairlet centres.
    """
    if method == "kmedoids":
        return kmedoids(fairlet_centers, k, random_state=random_state)
    if method == "kmedians":
        return k_medians(fairlet_centers, k, random_state=random_state)
    if method == "kmeans":
        return run_kmeans(fairlet_centers, k, random_state=random_state)
    raise ValueError(f"Unknown method '{method}'")
