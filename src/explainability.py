# explainability.py
# Rule-based and exemplar-based explainability for clustering results.
#
# Two methods from the thesis proposal:
#   1. Rule-based : Decision-tree surrogate trained on cluster labels
#                   → human-readable if-then rules (no sensitive attribute used)
#   2. Exemplar   : Medoid of each cluster = representative prototype
#
# References
# ----------
# Surrogate decision trees for clustering explainability:
#   Dasgupta et al. (2020). "Explainability via Decision Trees."
#   Moshkovitz et al. (2020). "Explainable k-Means and k-Medians Clustering."
#   ICML 2020. https://arxiv.org/abs/2002.12538
#
# Exemplar / medoid concept (K-Medoids / PAM):
#   Kaufman & Rousseeuw (1987). "Clustering by Means of Medoids."
#   In Statistical Data Analysis Based on the L1-Norm, pp. 405-416.
#   GitHub (scikit-learn-extra PAM): https://github.com/scikit-learn-contrib/scikit-learn-extra
#
# Fairlet decomposition (Twagner quadtree approach):
#   Chierichetti et al. (2017). "Fair Clustering Through Fairlets."
#   NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/978fce5bcc4501f762b2523a5f23b66c-Abstract.html
#   GitHub (reference implementation): https://github.com/MilkaLichtblau/Multinomial_Fairlets

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from src.config import TREE_MAX_DEPTH, TEST_SIZE, RANDOM_STATE


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based explainability (decision tree surrogate)
# ─────────────────────────────────────────────────────────────────────────────

def explainability_metrics(X, labels, feature_names=None, max_depth=None,
                           random_state=None, test_size=None,
                           sensitive_col_prefix="sensitive"):
    """
    Train a decision tree to mimic cluster labels and extract interpretability
    metrics.  The sensitive attribute columns are excluded from the tree to
    prevent the rules from depending on the protected group.

    Parameters
    ----------
    X                    : ndarray (n, d)
    labels               : ndarray (n,) – cluster assignments
    feature_names        : list of str or None
    max_depth            : int (default from config)
    random_state         : int (default from config)
    test_size            : float (default from config)
    sensitive_col_prefix : str – columns starting with this prefix are dropped
                           from the tree features (fairness-aware rules)

    Returns
    -------
    dict with keys:
        tree_fidelity   – accuracy of tree on held-out test split
        tree_depth      – actual depth of the fitted tree
        tree_leaves     – number of leaf nodes
        decision_rules  – human-readable rule string (top-level rules)
    """
    if max_depth    is None: max_depth    = TREE_MAX_DEPTH
    if random_state is None: random_state = RANDOM_STATE
    if test_size    is None: test_size    = TEST_SIZE

    X_use = X
    names = list(feature_names) if feature_names is not None else None

    # Drop sensitive attribute columns from the surrogate tree (fairness)
    if names is not None:
        keep = [i for i, n in enumerate(names)
                if not n.lower().startswith(sensitive_col_prefix.lower())]
        X_use = X[:, keep]
        names = [names[i] for i in keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, labels, test_size=test_size, random_state=random_state
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    fidelity = float(clf.score(X_test, y_test))

    # Human-readable rules (truncated to max_depth)
    rules = export_text(clf, feature_names=names, max_depth=max_depth)

    return {
        "tree_fidelity":  fidelity,
        "tree_depth":     clf.get_depth(),
        "tree_leaves":    clf.get_n_leaves(),
        "decision_rules": rules,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exemplar-based explainability (medoid prototypes)
# ─────────────────────────────────────────────────────────────────────────────

def exemplar_metrics(X, labels):
    """
    Find the medoid (most representative point) for each cluster and compute
    exemplar-based quality metrics.

    The medoid minimises the sum of L2 distances to all cluster members –
    it is the most centrally located actual data point in each cluster.

    Metrics
    -------
    avg_exemplar_coverage  : average fraction of cluster members whose
                             distance to their medoid is <= mean + 1*std
                             (higher = tighter, more representative exemplars)
    min_exemplar_coverage  : worst-cluster coverage (robustness indicator)
    avg_medoid_distance    : mean of per-cluster (mean distance from medoid to
                             all members) – measures cluster compactness around
                             the exemplar (lower = more compact)
    medoid_indices         : list of row indices of medoids per cluster

    Reference
    ---------
    Kaufman & Rousseeuw (1987) medoid concept.
    scikit-learn-extra PAM: https://github.com/scikit-learn-contrib/scikit-learn-extra
    """
    unique_labels = np.unique(labels)
    medoid_indices = []
    coverages = []
    medoid_dists = []

    for k in unique_labels:
        members_idx = np.where(labels == k)[0]
        members = X[members_idx]

        if len(members) == 1:
            medoid_indices.append(int(members_idx[0]))
            coverages.append(1.0)
            medoid_dists.append(0.0)
            continue

        D = cdist(members, members)
        best_local = int(np.argmin(D.sum(axis=1)))
        medoid_indices.append(int(members_idx[best_local]))

        # Distance from medoid to every cluster member
        dists_to_medoid = D[best_local]

        # Coverage: fraction within mean + 1*std distance to medoid
        threshold = dists_to_medoid.mean() + dists_to_medoid.std()
        coverage = float(np.mean(dists_to_medoid <= threshold))
        coverages.append(coverage)

        # Compactness: mean distance from medoid to members
        medoid_dists.append(float(dists_to_medoid.mean()))

    return {
        "avg_exemplar_coverage": float(np.mean(coverages)),
        "min_exemplar_coverage": float(np.min(coverages)),
        "avg_medoid_distance":   float(np.mean(medoid_dists)),
        "medoid_indices":        medoid_indices,
    }
