# explainability.py
# Rule-based and exemplar-based explainability for clustering results.
#
# Two methods from the thesis proposal:
#   1. Rule-based : Decision-tree surrogate trained on cluster labels
#                   → human-readable if-then rules (no sensitive attribute used)
#   2. Exemplar   : Medoid of each cluster = representative prototype
#
# Reference for surrogate decision trees in clustering explainability:
#   Dasgupta et al. (2020). "Explainability via Decision Trees."
#   Moshkovitz et al. (2020). "Explainable k-Means and k-Medians Clustering."
#   ICML 2020. https://arxiv.org/abs/2002.12538

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
    Find the medoid (most representative point) for each cluster.
    The medoid minimises the sum of distances to all cluster members.

    Metrics
    -------
    avg_exemplar_coverage : average fraction of cluster members within one
                            std-dev of their medoid (exemplar coverage)
    medoid_indices        : list of row indices of medoids per cluster

    Reference: Kaufman & Rousseeuw (1987) medoid concept.
    """
    unique_labels = np.unique(labels)
    medoid_indices = []
    coverages = []

    for k in unique_labels:
        members_idx = np.where(labels == k)[0]
        members = X[members_idx]

        if len(members) == 1:
            medoid_indices.append(int(members_idx[0]))
            coverages.append(1.0)
            continue

        D = cdist(members, members)
        best_local = int(np.argmin(D.sum(axis=1)))
        medoid_indices.append(int(members_idx[best_local]))

        # Coverage: fraction within mean + 1*std distance to medoid
        dists_to_medoid = D[best_local]
        threshold = dists_to_medoid.mean() + dists_to_medoid.std()
        coverage = float(np.mean(dists_to_medoid <= threshold))
        coverages.append(coverage)

    return {
        "avg_exemplar_coverage": float(np.mean(coverages)),
        "medoid_indices":        medoid_indices,
    }
