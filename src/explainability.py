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

# Clusters larger than this use approximate medoid search (subsample) to avoid
# building an n×n distance matrix that exhausts RAM on large datasets.
# Below the threshold the exact PAM algorithm is used (full cdist).
# Reference: CLARA algorithm (Kaufman & Rousseeuw 1990, §3) uses the same
# subsampling principle for scalable medoid search.
_MEDOID_EXACT_LIMIT = 2000


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

    # ── Tree for rules: trained on FULL dataset ──────────────────────────────
    # Matches plot_decision_tree() in visualization.py which also trains on
    # full data.  Identical data + same random_state → deterministic CART →
    # terminal rules and visual figure show exactly the same tree structure.
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_use, labels)

    # Human-readable rules (truncated to max_depth)
    rules = export_text(clf, feature_names=names, max_depth=max_depth)

    # ── Fidelity: evaluated on held-out test split (unbiased estimate) ───────
    # A separate eval tree is trained on the train split so fidelity reflects
    # generalisation ability, not in-sample training accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X_use, labels, test_size=test_size, random_state=random_state
    )
    clf_eval = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf_eval.fit(X_train, y_train)
    fidelity = float(clf_eval.score(X_test, y_test))

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
    the PAM cost as an exemplar quality metric.

    Medoid definition
    -----------------
    The medoid of cluster C_k is the actual data point that minimises the
    total distance to all other members:

        m_k = argmin   Σ   d(x, x')
              x ∈ C_k x'∈C_k

    PAM cost (published metric)
    ---------------------------
    The PAM cost is the total sum of distances from every point to its
    cluster medoid, normalised by the number of points so that datasets
    of different sizes are comparable:

        PAM cost = (1/n)  Σ   d( x, m_k(x) )
                         x∈X

    This is directly derived from the PAM objective function in
    Kaufman & Rousseeuw (1987):
        "Clustering by Medoids (PAM)", Chapter 2 in
        Finding Groups in Data: An Introduction to Cluster Analysis.
        Wiley, New York.

    A lower PAM cost means points are closer to their exemplar →
    the exemplar is more representative of its cluster.

    Large-cluster handling (CLARA-style subsampling)
    -------------------------------------------------
    Clusters with more than _MEDOID_EXACT_LIMIT members use an approximate
    medoid search: a random subsample of size _MEDOID_EXACT_LIMIT is drawn,
    the medoid of the subsample is found exactly, then distances from that
    one medoid row to ALL cluster members are computed (shape 1×n, not n×n).
    This avoids building an n×n distance matrix that exhausts RAM on large
    datasets (e.g. bank: 40 000 rows).
    Reference: Kaufman & Rousseeuw (1990) CLARA algorithm §3.

    Parameters
    ----------
    X      : ndarray (n, d)  — standardised feature matrix
    labels : ndarray (n,)    — cluster assignment for each point

    Returns
    -------
    dict with keys:
        pam_cost             : float  — (1/n) Σ d(x, m_k(x))  [published]
        medoid_indices       : list   — row indices of medoids per cluster
        avg_exemplar_coverage: float  — kept in CSV only, not displayed
        min_exemplar_coverage: float  — kept in CSV only, not displayed
        avg_medoid_distance  : float  — legacy name, equals pam_cost
    """
    unique_labels = np.unique(labels)
    n = len(X)
    medoid_indices  = []
    coverages       = []
    total_pam_cost  = 0.0   # accumulates Σ d(x, m_k) across all clusters

    for k in unique_labels:
        members_idx = np.where(labels == k)[0]
        members     = X[members_idx]

        if len(members) == 1:
            medoid_indices.append(int(members_idx[0]))
            coverages.append(1.0)
            # Single-point cluster: distance to its own medoid is 0
            continue

        if len(members) <= _MEDOID_EXACT_LIMIT:
            # ── Exact PAM: full n×n pairwise distance matrix ─────────────
            D          = cdist(members, members)
            best_local = int(np.argmin(D.sum(axis=1)))
            # Distances from medoid to every cluster member (already computed)
            dists_to_medoid = D[best_local]

        else:
            # ── Approximate PAM for large clusters (CLARA-style subsample) ──
            # Step 1: draw a random subsample to find the candidate medoid.
            #         cdist on the subsample is at most _MEDOID_EXACT_LIMIT²
            #         entries — safe even for 40 000-row datasets.
            # Step 2: compute distances from that ONE medoid row to ALL members
            #         (shape 1×n, not n×n — ~n× less memory).
            # Reference: Kaufman & Rousseeuw (1990) CLARA algorithm §3.
            rng       = np.random.default_rng(RANDOM_STATE)
            sub_idx   = rng.choice(len(members), _MEDOID_EXACT_LIMIT,
                                   replace=False)
            sub       = members[sub_idx]
            D_sub     = cdist(sub, sub)
            best_sub  = int(np.argmin(D_sub.sum(axis=1)))
            best_local = sub_idx[best_sub]   # index into full `members` array

            # Exact distances from the approximate medoid to ALL cluster members
            # Shape: (1, n_members) — never larger than 1 × cluster_size
            dists_to_medoid = cdist(members[[best_local]], members)[0]

        medoid_indices.append(int(members_idx[best_local]))

        # Accumulate into total PAM cost: Σ d(x, m_k) for this cluster
        total_pam_cost += float(dists_to_medoid.sum())

        # Coverage: fraction within mean + 1*std distance to medoid
        # (kept for CSV reference only — not shown in terminal table)
        threshold = dists_to_medoid.mean() + dists_to_medoid.std()
        coverages.append(float(np.mean(dists_to_medoid <= threshold)))

    # PAM cost normalised by n so it is scale-independent across datasets
    # Formula: (1/n) Σ_{x∈X} d(x, m_{k(x)})
    pam_cost = total_pam_cost / n if n > 0 else 0.0

    return {
        "pam_cost":              pam_cost,           # published PAM cost ÷ n
        "avg_medoid_distance":   pam_cost,            # legacy alias (same value)
        "avg_exemplar_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "min_exemplar_coverage": float(np.min(coverages))  if coverages else 0.0,
        "medoid_indices":        medoid_indices,
    }
