# fairness.py
# Fair clustering via Twagner quadtree-based fairlet decomposition.
#
# Primary reference:
#   Chierichetti, F., Kumar, R., Lattanzi, S., & Vassilvitskii, S. (2017).
#   "Fair Clustering Through Fairlets."
#   Advances in Neural Information Processing Systems (NeurIPS 2017).
#   https://proceedings.neurips.cc/paper/2017/hash/
#       978fce5bcc4501f762b2523a5f23b66c-Abstract.html
#
# Implementation follows Algorithm 1 (quadtree construction) and Lemma 3
# (basic fairlet decomposition with three-phase leftover bubbling) from
# the paper above, as implemented in the thesis notebook
# (fair_explainable_clustering_v2.ipynb).

import numpy as np
from scipy.spatial.distance import cdist
from src.config import QUADTREE_MAX_LEVELS, QUADTREE_RANDOM_SHIFT, QUADTREE_EPSILON


# ─────────────────────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────────────────────

class TreeNode:
    """Node in the quadtree used for fairlet decomposition."""

    __slots__ = ("indices", "reds", "blues", "children")

    def __init__(self):
        self.indices = []    # original dataset row indices in this cell
        self.reds = []       # indices of sensitive==1 points
        self.blues = []      # indices of sensitive==0 points
        self.children = []

    def populate_colors(self, colors):
        """Recursively collect red/blue indices from leaves up to this node."""
        if not self.children:
            for idx in self.indices:
                if colors[idx] == 1:
                    self.reds.append(idx)
                else:
                    self.blues.append(idx)
        else:
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)


# ─────────────────────────────────────────────────────────────────────────────
# Quadtree construction  (Algorithm 1, Chierichetti et al. 2017)
# ─────────────────────────────────────────────────────────────────────────────

def _build_quadtree_recursive(dataset, indices, lower, upper, level, max_levels, epsilon):
    """Recursively partition the bounding box, assigning points to leaf nodes."""
    node = TreeNode()
    node.indices = list(indices)

    widths = upper - lower
    # Stop if cell is tiny, no points, or max depth reached
    if (max_levels > 0 and level >= max_levels) or len(indices) <= 1 or np.max(widths) < epsilon:
        return node

    # Split along the widest dimension at the midpoint
    dim = int(np.argmax(widths))
    mid = (lower[dim] + upper[dim]) / 2.0

    left_idx = [i for i in indices if dataset[i, dim] <= mid]
    right_idx = [i for i in indices if dataset[i, dim] > mid]

    if not left_idx or not right_idx:
        return node   # can't split further

    lower_right = lower.copy(); lower_right[dim] = mid
    upper_left = upper.copy(); upper_left[dim] = mid

    node.children = [
        _build_quadtree_recursive(dataset, left_idx,  lower, upper_left,  level + 1, max_levels, epsilon),
        _build_quadtree_recursive(dataset, right_idx, lower_right, upper,  level + 1, max_levels, epsilon),
    ]
    node.indices = []   # interior nodes don't store indices directly
    return node


def build_quadtree(dataset, max_levels=0, random_shift=True, epsilon=QUADTREE_EPSILON):
    """
    Build a quadtree over `dataset` (n_samples × n_features).

    Parameters
    ----------
    dataset      : ndarray (n, d) – pre-processed feature matrix
    max_levels   : int – 0 means fully recursive (no depth limit)
    random_shift : bool – random translation of bounding box to reduce
                   worst-case behaviour (Theorem 3.2, Chierichetti et al.)
    epsilon      : float – minimum cell width before stopping

    Returns
    -------
    root : TreeNode
    """
    n, d = dataset.shape
    lower = dataset.min(axis=0).copy()
    upper = dataset.max(axis=0).copy()

    # Pad bounding box slightly
    widths = upper - lower
    lower -= 0.01 * widths
    upper += 0.01 * widths

    if random_shift:
        rng = np.random.default_rng(42)
        shift = rng.uniform(0, upper - lower)
        lower += shift
        upper += shift

    return _build_quadtree_recursive(
        dataset, list(range(n)), lower, upper, 0, max_levels, epsilon
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fairlet helpers
# ─────────────────────────────────────────────────────────────────────────────

def _balanced(p, q, r, b):
    """
    (p,q)-balance condition from Chierichetti et al. Definition 2.1.
    A set with r reds and b blues is balanced if min(r/b, b/r) >= p/q.
    """
    if r == 0 or b == 0:
        return False
    return min(r / b, b / r) >= (p / q)


def _make_fairlet(points, dataset):
    """
    Create one fairlet from a list of point indices.
    Medoid = point minimising total L2 distance to all fairlet members.
    Returns (fairlet_indices, medoid_index, intra_cost).
    """
    pts = np.array(points)
    if len(pts) == 1:
        return list(pts), pts[0], 0.0

    sub = dataset[pts]
    D = cdist(sub, sub)
    total = D.sum(axis=1)
    best_local = int(np.argmin(total))
    medoid = int(pts[best_local])
    cost = float(total[best_local])
    return list(pts), medoid, cost


def _basic_fairlet_decomposition(p, q, blues, reds, dataset, fairlets, centers):
    """
    Lemma 3 from Chierichetti et al.:
    Decompose 'reds' and 'blues' into (p,q)-balanced fairlets with
    three-phase leftover handling so NO points are ever dropped.

    Returns (leftover_blues, leftover_reds) – excess after complete fairlets.
    """
    blues = list(blues)
    reds = list(reds)

    # Form complete (p,q)-balanced groups: q reds + p blues per fairlet
    while len(reds) >= q and len(blues) >= p:
        group = [reds.pop() for _ in range(q)] + [blues.pop() for _ in range(p)]
        pts, med, cost = _make_fairlet(group, dataset)
        fairlets.append(pts)
        centers.append(med)

    return blues, reds   # leftovers bubbled up to parent


# ─────────────────────────────────────────────────────────────────────────────
# Twagner tree-based fairlet decomposition  (Algorithm 2, with bubbling)
# ─────────────────────────────────────────────────────────────────────────────

def _decompose_node(node, p, q, dataset, fairlets, centers):
    """
    Post-order traversal: decompose children first, then handle this node's
    points + leftovers bubbled up from children.

    Returns (leftover_blues, leftover_reds) for the parent.
    """
    accumulated_blues = list(node.blues if node.children == [] else [])
    accumulated_reds  = list(node.reds  if node.children == [] else [])

    # Recurse into children and collect their leftovers
    for child in node.children:
        lb, lr = _decompose_node(child, p, q, dataset, fairlets, centers)
        accumulated_blues.extend(lb)
        accumulated_reds.extend(lr)

    # Try to form fairlets from everything we have at this level
    leftover_blues, leftover_reds = _basic_fairlet_decomposition(
        p, q, accumulated_blues, accumulated_reds, dataset, fairlets, centers
    )
    return leftover_blues, leftover_reds


def twagner_fairlet_decomposition(dataset, sensitive, p=1, q=2):
    """
    Full Twagner quadtree fairlet decomposition.

    Steps (follows Chierichetti et al. NIPS 2017):
      1. Build quadtree with optional random shift.
      2. Populate red/blue lists on each node bottom-up.
      3. Post-order traversal: form (p,q)-balanced fairlets greedily;
         leftover points bubble up to parent (no points dropped).
      4. Any global leftover after root is paired 1-1 into mixed fairlets.

    Parameters
    ----------
    dataset   : ndarray (n, d)
    sensitive : ndarray (n,) – binary, 1=minority group
    p, q      : int – balance parameters (p/q = min allowed minority ratio)

    Returns
    -------
    fairlets : list of lists of int  – each inner list is one fairlet
    centers  : list of int           – medoid index per fairlet
    """
    root = build_quadtree(
        dataset,
        max_levels=QUADTREE_MAX_LEVELS,
        random_shift=QUADTREE_RANDOM_SHIFT,
    )
    root.populate_colors(sensitive)

    fairlets = []
    centers  = []

    leftover_blues, leftover_reds = _decompose_node(root, p, q, dataset, fairlets, centers)

    # Pair up any remaining leftovers (cannot form balanced groups alone)
    while leftover_blues and leftover_reds:
        group = [leftover_blues.pop(), leftover_reds.pop()]
        pts, med, _ = _make_fairlet(group, dataset)
        fairlets.append(pts)
        centers.append(med)

    # Remaining single-colour points: append as singletons
    for idx in leftover_blues + leftover_reds:
        fairlets.append([idx])
        centers.append(idx)

    return fairlets, centers


# ─────────────────────────────────────────────────────────────────────────────
# Label propagation: fairlet labels -> original point labels
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels_from_fairlets(fairlets, center_cluster_labels, n):
    """
    Map cluster labels (assigned to fairlet centres) back to all original points.

    Parameters
    ----------
    fairlets              : list of lists of int
    center_cluster_labels : ndarray – cluster label for each fairlet centre
    n                     : int – total number of original data points

    Returns
    -------
    labels : ndarray (n,)
    """
    labels = np.zeros(n, dtype=int)
    for fairlet, lab in zip(fairlets, center_cluster_labels):
        for idx in fairlet:
            labels[idx] = lab
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Fairness evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def fairness_metrics(labels, sensitive, p=1, q=2):
    """
    Compute fairness metrics for a clustering result.

    Metrics
    -------
    min_balance    : minimum (p,q)-balance across all clusters
                     (worst-case cluster – key metric from Chierichetti et al.)
    avg_balance    : average balance across clusters
    violation_rate : fraction of clusters violating the p/q balance threshold
    avg_dp_gap     : average absolute demographic parity gap per cluster

    Balance of a cluster = min(n_red/n_blue, n_blue/n_red)
    A cluster violates balance if balance < p/q.
    """
    K = len(np.unique(labels))
    balances = []
    violations = 0
    dp_gaps = []
    global_ratio = float(sensitive.mean())

    for k in range(K):
        mask = labels == k
        group = sensitive[mask]
        n0 = int(np.sum(group == 0))
        n1 = int(np.sum(group == 1))

        if n0 + n1 == 0:
            continue

        # Balance
        if n0 == 0 or n1 == 0:
            bal = 0.0
        else:
            bal = min(n0, n1) / max(n0, n1)
        balances.append(bal)

        if bal < (p / q):
            violations += 1

        # Demographic parity gap
        dp_gaps.append(abs(group.mean() - global_ratio))

    return {
        "min_balance":    float(np.min(balances))  if balances else 0.0,
        "avg_balance":    float(np.mean(balances)) if balances else 0.0,
        "violation_rate": violations / K,
        "avg_dp_gap":     float(np.mean(dp_gaps))  if dp_gaps  else 0.0,
    }
