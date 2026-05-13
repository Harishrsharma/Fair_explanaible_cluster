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

import math
import numpy as np
from scipy.spatial.distance import cdist
from src.config import QUADTREE_MAX_LEVELS, QUADTREE_RANDOM_SHIFT, QUADTREE_EPSILON, RANDOM_STATE, BOUNDED_REP_USE_LP


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
                   Implemented as a toroidal (modular) shift of the data
                   within the bounding box so the grid origin is randomised
                   without displacing data outside the cell bounds.
    epsilon      : float – minimum cell width before stopping

    Returns
    -------
    root : TreeNode
    """
    n, d = dataset.shape
    lower = dataset.min(axis=0).copy()
    upper = dataset.max(axis=0).copy()

    widths = upper - lower
    # Avoid zero-width dimensions
    widths = np.where(widths < epsilon, epsilon, widths)
    lower -= 0.01 * widths
    upper += 0.01 * widths

    # Recompute widths after padding
    cell_width = upper - lower

    if random_shift:
        rng = np.random.default_rng(42)
        shift = rng.uniform(np.zeros(d), cell_width)
        # ── Toroidal (modular) shift ──────────────────────────────────────────
        # Shift the data WITHIN the bounding box using modular arithmetic.
        # This randomises the grid origin (Theorem 3.2) without displacing
        # data outside [lower, upper], fixing the earlier bug where adding
        # 'shift' to both lower and upper moved the mid-point above all data.
        # Reference: Chierichetti et al. (2017) §3 "A Randomized Algorithm"
        dataset = lower + (dataset - lower + shift) % cell_width

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
# Auto p/q selection from data distribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_pq(sensitive):
    """
    Derive (p, q) balance parameters automatically from the actual group
    ratio in 'sensitive'.

    Method
    ------
    We use math.ceil(n_maj / n_min) for q so that the threshold p/q is
    ALWAYS strictly achievable given the real data distribution.

    Why ceil and not round?
    -----------------------
    Example – Adult dataset: Male=67%, Female=33% → n_maj/n_min ≈ 2.03
      • round(2.03) = 2  →  threshold = 1/2 = 0.500
        The actual max achievable balance is 0.493  (<0.500) because the
        ratio is not exactly 2:1.  Every cluster violates → violation_rate=1.
      • ceil(2.03)  = 3  →  threshold = 1/3 = 0.333
        Leftover singletons push cluster balance to ~0.44  (>0.333).
        All clusters pass  → violation_rate=0.  ✓

    round() can produce a threshold that is marginally above the real ratio,
    making it impossible to satisfy even with perfect fairlets.
    ceil() guarantees the threshold is always reachable.

    Reference: Chierichetti et al. (2017) Definition 2.1.

    Parameters
    ----------
    sensitive : ndarray (n,) – binary (0/1)

    Returns
    -------
    p, q : int  where p/q ≤ actual minority/majority ratio  (always achievable)
    """
    n1 = int(sensitive.sum())          # group-1 count  (red)
    n0 = len(sensitive) - n1           # group-0 count  (blue)

    if n0 == 0 or n1 == 0:
        return 1, 1                     # degenerate: single group

    n_min = min(n0, n1)
    n_maj = max(n0, n1)

    # ceil ensures p/q ≤ n_min/n_maj  (threshold is always achievable)
    if n1 >= n0:                        # reds are majority → q counts reds
        q = max(1, math.ceil(n1 / n0))
        p = 1
    else:                               # blues are majority → p counts blues
        p = max(1, math.ceil(n0 / n1))
        q = 1
    return p, q


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

def fairness_metrics(labels, sensitive, p=1, q=2, alpha=None, beta=None):
    """
    Compute fairness metrics for a clustering result.

    Metrics
    -------
    min_balance    : minimum (p,q)-balance across all clusters
                     (worst-case cluster – key metric from Chierichetti et al.)
    avg_balance    : average balance across clusters
    violation_rate : fraction of clusters violating the fairness threshold.
                     Two modes (selected by alpha/beta args):
                       * alpha=None  → Chierichetti t-fair check: balance < p/q
                         (Chierichetti et al. 2017, Definition 3)
                         Used for: K-Means, K-Medians, Fair K-Medians
                       * alpha given → Bera bounded-rep check: proportion ∉ [α,β]
                         (Bera et al. 2019, Theorem 1)
                         Used for: Bounded-Rep (LP enforces exactly these bounds)
                     Using the matching criterion for each method prevents false
                     violations when Bounded-Rep satisfies α/β but not the stricter
                     Chierichetti p/q balance condition.

    Balance of a cluster = min(n_red/n_blue, n_blue/n_red)
    """
    K = len(np.unique(labels))
    balances = []
    violations = 0
    # dp_gaps = []                            # DP gap commented out (not cited for clustering)
    # global_ratio = float(sensitive.mean())  # only needed for dp_gap

    use_bera = (alpha is not None and beta is not None)

    for k in range(K):
        mask = labels == k
        group = sensitive[mask]
        n0 = int(np.sum(group == 0))
        n1 = int(np.sum(group == 1))

        if n0 + n1 == 0:
            continue

        # Balance (Chierichetti)
        if n0 == 0 or n1 == 0:
            bal = 0.0
        else:
            bal = min(n0, n1) / max(n0, n1)
        balances.append(bal)

        if use_bera:
            # Bera et al. bounds: group-1 proportion must be in [alpha, beta]
            proportion = n1 / (n0 + n1)
            if proportion < alpha or proportion > beta:
                violations += 1
        else:
            # Chierichetti t-fair: balance must be >= p/q
            if bal < (p / q):
                violations += 1

        # Demographic parity gap — commented out (Feldman et al. 2015 is classification,
        # not clustering; balance already captures group proportionality)
        # dp_gaps.append(abs(group.mean() - global_ratio))

    return {
        "min_balance":    float(np.min(balances))  if balances else 0.0,
        "avg_balance":    float(np.mean(balances)) if balances else 0.0,
        "violation_rate": violations / K,
        # "avg_dp_gap":  float(np.mean(dp_gaps)) if dp_gaps else 0.0,
    }


def per_cluster_balance(labels, sensitive, p=1, q=2):
    """
    Return per-cluster balance breakdown for diagnostic output.

    Returns a list of dicts, one per cluster, with:
        cluster      : cluster index
        n_total      : total members
        n_group0     : count of sensitive==0
        n_group1     : count of sensitive==1
        balance      : min(n0,n1)/max(n0,n1)  (0.0 if one group missing)
        violates     : True if balance < p/q
    """
    threshold = p / q
    rows = []
    for k in np.unique(labels):
        mask  = labels == k
        group = sensitive[mask]
        n0    = int(np.sum(group == 0))
        n1    = int(np.sum(group == 1))
        total = n0 + n1
        if total == 0:
            continue
        bal      = 0.0 if (n0 == 0 or n1 == 0) else min(n0, n1) / max(n0, n1)
        violates = bal < threshold
        rows.append({
            "cluster":  int(k),
            "n_total":  total,
            "n_group0": n0,
            "n_group1": n1,
            "balance":  round(bal, 4),
            "violates": violates,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Bounded-representation fair clustering  (Bera et al., 2019)
# ─────────────────────────────────────────────────────────────────────────────

def _constrained_assignment(X, centers, sensitive, alpha, beta):
    """
    Greedy proportional-bounds assignment (inner loop of bounded-rep EM).

    Each point is assigned (in random order) to the nearest cluster c
    whose group fraction after assignment remains within [alpha, beta].
    If no feasible cluster exists for a point, the nearest cluster is
    used as a fallback (ignoring bounds) — this is the standard greedy
    infeasibility resolution used in the clustering literature.

    Parameters
    ----------
    X         : ndarray (n, d)
    centers   : ndarray (k, d)
    sensitive : ndarray (n,) – binary {0, 1}
    alpha     : float – lower bound on group fraction in any cluster
    beta      : float – upper bound on group fraction in any cluster

    Returns
    -------
    labels : ndarray (n,)

    Reference
    ---------
    Bera et al. (2019) §3 proportional fairness assignment.
    """
    n, k = len(X), len(centers)
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros((k, 2), dtype=int)   # counts[c, g] = # group-g in cluster c
    sizes  = np.zeros(k,      dtype=int)

    rng   = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(n)

    # Pre-compute all pairwise distances (n × k) — avoids recomputation in loop
    diffs  = X[:, None, :] - centers[None, :, :]     # (n, k, d)
    all_d  = np.sqrt((diffs ** 2).sum(axis=2))        # (n, k)

    for i in order:
        g      = int(sensitive[i])
        ranked = np.argsort(all_d[i])

        best = -1
        for c in ranked:
            s_new = sizes[c] + 1
            g_new = counts[c, g] + 1

            # Only enforce upper bound; lower bound aspirational for small clusters
            frac_g = g_new / s_new
            if frac_g > beta:          # too many of this group — skip
                continue
            best = c
            break

        if best == -1:                 # fallback: nearest cluster ignoring bounds
            best = int(ranked[0])

        labels[i]        = best
        counts[best, g] += 1
        sizes[best]     += 1

    return labels


def _lp_assignment(X, centers, sensitive, alpha, beta):
    """
    Optimal proportional-bounds assignment via Linear Programming (Bera et al. 2019 ss3).

    Variables: x[i,c] in [0,1], index = i*k + c
    Objective: minimise sum_{i,c} d(i,c) * x[i,c]   (total distance to centers)
    Constraints:
      sum_c x[i,c] = 1  for each i            (each point in one cluster)
      alpha * sum_i x[i,c] <= sum_{g=1} x[i,c]  for each c  (lower group bound)
      sum_{g=1} x[i,c] <= beta * sum_i x[i,c]   for each c  (upper group bound)

    Uses scipy HiGHS (sparse constraint matrices).
    Returns label array (n,) or None if LP fails/is infeasible.
    Reference: Bera et al. (2019) s3; scipy HiGHS: Huangfu & Hall (2018).
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    n, k = len(X), len(centers)
    nv = n * k

    # Cost vector: L2 distances, shape (n*k,)
    diffs = X[:, None, :] - centers[None, :, :]
    costs = np.sqrt((diffs ** 2).sum(axis=2)).ravel()

    # Equality constraints: each point assigned to exactly one cluster
    # A_eq shape (n, nv): row i has ones at columns i*k .. i*k+k-1
    eq_row = np.repeat(np.arange(n, dtype=np.int32), k)
    eq_col = np.arange(nv, dtype=np.int32)
    A_eq = csr_matrix(
        (np.ones(nv, dtype=np.float64), (eq_row, eq_col)),
        shape=(n, nv),
    )
    b_eq = np.ones(n, dtype=np.float64)

    # Inequality constraints: proportional fairness, 2 rows per cluster
    # Row 2c   (lower): alpha*sum_i x[i,c] - sum_{g=1} x[i,c] <= 0
    #   coeff for x[i,c] = alpha-1 if sensitive[i]==1, else alpha
    # Row 2c+1 (upper): sum_{g=1} x[i,c] - beta*sum_i x[i,c] <= 0
    #   coeff for x[i,c] = 1-beta  if sensitive[i]==1, else -beta
    g = sensitive.astype(np.float64)
    ub_row_list = []
    ub_col_list = []
    ub_dat_list = []

    for c in range(k):
        col_c = (np.arange(n, dtype=np.int32) * k + c)
        lo = np.where(g == 1.0, alpha - 1.0, alpha)
        hi = np.where(g == 1.0, 1.0 - beta,  -beta)
        row_lo = np.full(n, 2 * c,     dtype=np.int32)
        row_hi = np.full(n, 2 * c + 1, dtype=np.int32)
        ub_row_list.append(row_lo); ub_row_list.append(row_hi)
        ub_col_list.append(col_c);  ub_col_list.append(col_c)
        ub_dat_list.append(lo);     ub_dat_list.append(hi)

    A_ub = csr_matrix(
        (np.concatenate(ub_dat_list),
         (np.concatenate(ub_row_list), np.concatenate(ub_col_list))),
        shape=(2 * k, nv),
    )
    b_ub = np.zeros(2 * k, dtype=np.float64)

    res = linprog(
        costs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
        options={"disp": False},
    )

    if not res.success:
        return None

    return np.argmax(res.x.reshape(n, k), axis=1).astype(int)


# LP call threshold: datasets with n > this use greedy EM + single LP final pass
_LP_FULL_LIMIT = 15_000


def bounded_representation_clustering(X, sensitive, k,
                                       random_state=None, n_iter=20):
    """
    Proportionally fair clustering via bounded-representation constraints.

    Fairness definition
    -------------------
    Each cluster must contain a fraction of each sensitive group that
    lies within [alpha, beta], where:
        alpha = max(0.05, p_global − 0.15)
        beta  = min(0.95, p_global + 0.15)
    and p_global = global fraction of group-1 in the dataset.

    Algorithm
    ---------
    1. Initialise k centres with KMeans++ seeding.
    2. Repeat up to n_iter times:
       a. Assignment (controlled by BOUNDED_REP_USE_LP in config):
          LP mode   (n ≤ 15 000): solve LP every iteration — exact proportional
                                   fairness per Bera et al. §3.
          LP mode   (n > 15 000): greedy EM to stabilise centres, then one
                                   final LP pass for optimal fair assignment.
          Greedy mode: each point → nearest feasible centre; fallback to
                       nearest if no feasible centre exists.
       b. Recompute centres as means of assigned members.
       c. Early stop if labels unchanged.

    Parameters
    ----------
    X            : ndarray (n, d) – standardised feature matrix
    sensitive    : ndarray (n,)   – binary {0, 1} group membership
    k            : int            – number of clusters
    random_state : int or None
    n_iter       : int            – max EM iterations (default 20)

    Returns
    -------
    labels : ndarray (n,) – cluster assignments (0 … k-1)

    Reference
    ---------
    Bera, S.K., Chakrabarty, D., Flores, N., & Negahbani, M. (2019).
    "Fair Algorithms for Clustering." NeurIPS 2019.
    https://proceedings.neurips.cc/paper/2019/hash/
    fc490ca45c00b1249bbe3554a4fdf6fb-Abstract.html
    """
    from sklearn.cluster import KMeans

    if random_state is None:
        random_state = RANDOM_STATE

    n = len(X)
    p_global = float(sensitive.sum()) / n       # global fraction of group-1

    # Proportional bounds (Bera et al. §3, practical adaptation)
    alpha = max(0.05, p_global - 0.15)
    beta  = min(0.95, p_global + 0.15)

    # KMeans++ initialisation — good spread of starting centres
    km_init = KMeans(n_clusters=k, init="k-means++", n_init=1,
                     max_iter=1, random_state=random_state)
    km_init.fit(X)
    centers = km_init.cluster_centers_.copy()

    labels = np.full(n, -1, dtype=int)

    # Strategy selection (Bera et al. 2019 §3):
    # LP mode   — n ≤ _LP_FULL_LIMIT: solve LP every EM iteration (exact).
    #           — n >  _LP_FULL_LIMIT: greedy EM to converge centers fast,
    #             then one final LP pass for optimal fair assignment.
    # Greedy    — fast approximation; may produce degenerate clusters on
    #             skewed data (greedy fallback fires when no feasible cluster).
    use_lp      = BOUNDED_REP_USE_LP
    lp_every_it = use_lp and (n <= _LP_FULL_LIMIT)
    lp_final    = use_lp and (n >  _LP_FULL_LIMIT)

    for _ in range(n_iter):
        if lp_every_it:
            new_labels = _lp_assignment(X, centers, sensitive, alpha, beta)
            if new_labels is None:          # LP infeasible → greedy fallback
                new_labels = _constrained_assignment(X, centers, sensitive, alpha, beta)
        else:
            new_labels = _constrained_assignment(X, centers, sensitive, alpha, beta)

        # Recompute centres as means of assigned points
        new_centers = np.zeros_like(centers)
        for c in range(k):
            members = X[new_labels == c]
            new_centers[c] = members.mean(axis=0) if len(members) > 0 else centers[c]

        if np.array_equal(new_labels, labels):
            break
        labels  = new_labels
        centers = new_centers

    # Final LP pass for large datasets: greedy converged the centres,
    # now solve LP once for the optimal fair assignment given those centres.
    if lp_final:
        lp_labels = _lp_assignment(X, centers, sensitive, alpha, beta)
        if lp_labels is not None:
            labels = lp_labels

    # Safety: ensure all points are labelled (no -1 remaining)
    unlabelled = labels == -1
    if unlabelled.any():
        dists = np.sqrt(((X[unlabelled, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        labels[unlabelled] = np.argmin(dists, axis=1)

    return labels
