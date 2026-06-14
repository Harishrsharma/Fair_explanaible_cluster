# visualization.py
# Plotting helpers for the thesis pipeline.
#
# This module is ADDITIVE: it consumes the outputs of the existing
# pipeline (labels, X, results DataFrames) and produces three families
# of figures requested by the supervisor:
#
#   1. plot_metric_comparison(...)
#         One bar chart per metric across all clustering algorithms
#         (kmeans_baseline, kmedians_baseline, fair_kmedians, ...).
#
#   2. plot_feature_importance_per_cluster(...)
#         Top-5 most important features per cluster, one figure per
#         algorithm. Importance is derived from a one-vs-rest Random
#         Forest classifier trained to predict cluster membership.
#
#   3. plot_cluster_scatter(...)
#         2-D PCA scatter plot of the data coloured by cluster label,
#         one figure per algorithm.
#
# Naming convention for saved files:
#   results/figures/<dataset>/metric_<metric>.png
#   results/figures/<dataset>/feature_importance_<algorithm>.png
#   results/figures/<dataset>/scatter_<algorithm>.png
#
# Nothing in the existing modules is modified.

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / notebook usage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.config import (
    RANDOM_STATE, DATASETS, RESULTS_DIR,
    K_RANGE, DATASET_K_OVERRIDE,
)
from src.data_processing import load_dataset
from src.clustering import (
    run_kmeans, k_medians, cluster_fairlet_centers, find_optimal_k,
)
from src.fairness import (
    twagner_fairlet_decomposition, assign_labels_from_fairlets,
    fairness_metrics, compute_pq, bounded_representation_clustering,
)
from src.metrics import clustering_metrics
from src.explainability import explainability_metrics, exemplar_metrics


# Default folder for all generated figures (results/figures/<dataset>/...)
FIG_ROOT = os.path.join(RESULTS_DIR, "figures")


# ─────────────────────────────────────────────────────────────────────────────
# Display-friendly metric labels
# METRIC_LABELS  : used in chart *titles* — explains what the metric means
#                  and why it matters (professor requirement)
# METRIC_YLABELS : concise y-axis label (unit + direction) — no formulas
# METRIC_THRESHOLDS : (value, label) for a dotted reference line on the chart
# ─────────────────────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "silhouette":
        "Cluster Separation Quality\n"
        "How well-separated clusters are from each other"
        " — higher means clusters are more distinct (range: −1 to +1)",
    "silhouette_pca":
        "Cluster Separation Quality (PCA-reduced space)\n"
        "Same as Silhouette but computed after reducing features to 50 PCA components"
        " — corrects for high-dimensional distortion",
    "davies_bouldin":
        "Cluster Compactness vs. Separation — Davies–Bouldin Index\n"
        "Average ratio of within-cluster spread to between-cluster distance"
        " — lower means clusters are tighter and more distinct",
    "sse":
        "Cluster Tightness — Sum of Squared Errors\n"
        "Total squared distance of every point from its cluster centre"
        " — lower means points sit closer to their assigned cluster",
    "min_balance":
        "Worst-Case Fairness Balance\n"
        "Minority-to-majority ratio in the least balanced cluster"
        " — higher means no single cluster is severely dominated by one group"
        " (threshold line = minimum required balance p/q)",
    "avg_balance":
        "Average Fairness Balance Across Clusters\n"
        "Mean minority-to-majority ratio across all clusters"
        " — higher means groups are more evenly distributed overall",
    "min_bera_balance":
        "Worst-Case Proportional Fairness (Bera et al. 2019)\n"
        "How closely the worst cluster reflects the overall group proportion"
        " — higher means cluster composition is closer to the global ratio",
    "avg_bera_balance":
        "Average Proportional Fairness (Bera et al. 2019)\n"
        "Mean deviation of cluster compositions from the global group ratio"
        " — higher means clusters collectively reflect the population distribution",
    "violation_rate":
        "Fairness Violation Rate\n"
        "Fraction of clusters that breach the required fairness threshold"
        " — lower is better; 0.0 means all clusters satisfy the fairness constraint",
    # "avg_dp_gap": commented out — DP gap is a classification metric (Feldman 2015)
    "tree_fidelity":
        "Explanation Accuracy — Surrogate Tree Fidelity\n"
        "Fraction of data points whose cluster label is correctly predicted"
        " by a simple decision tree — higher means the rules reliably explain"
        " the clustering",
    "tree_depth":
        "Explanation Complexity — Surrogate Tree Depth\n"
        "Number of decision levels in the surrogate tree"
        " — lower means simpler, easier-to-read rules",
    "tree_leaves":
        "Explanation Complexity — Surrogate Tree Leaf Count\n"
        "Number of distinct if-then rules produced by the surrogate tree"
        " — lower means fewer rules to understand",
    "pam_cost":
        "Cluster Compactness — Average Distance to Representative\n"
        "Mean distance from each point to the closest cluster representative (medoid)"
        " — lower means the representative point better captures its cluster",
    "cost_of_fairness":
        "Price of Fairness — Quality Loss from Fairness Constraints\n"
        "Ratio of fair-algorithm compactness to baseline (K-Means) compactness"
        " — 1.0 means no quality loss; above 1.0 means fairness costs cluster quality",
    "avg_exemplar_coverage":
        "Average Exemplar Coverage\n"
        "Fraction of cluster members well-represented by their medoid"
        " — higher means the representative point captures more of its group",
    "min_exemplar_coverage":
        "Worst-Case Exemplar Coverage\n"
        "Lowest exemplar coverage across all clusters"
        " — higher means even the hardest-to-represent cluster has a good exemplar",
    "avg_medoid_distance":
        "Average Medoid Distance\n"
        "Mean distance from all points to their cluster medoid"
        " — lower means clusters are tighter around their representative",
    "runtime_s":
        "Algorithm Runtime (seconds)\n"
        "Wall-clock time to run the clustering — lower means faster",
}

METRIC_YLABELS = {
    "silhouette":            "Cluster separation  (higher = better)",
    "silhouette_pca":        "Cluster separation in reduced space  (higher = better)",
    "davies_bouldin":        "Cluster overlap  (lower = better)",
    "sse":                   "Total cluster spread  (lower = tighter)",
    "min_balance":           "Group balance in worst cluster  (higher = fairer)",
    "avg_balance":           "Average group balance  (higher = fairer)",
    "min_bera_balance":      "Worst cluster match to population mix  (higher = fairer)",
    "avg_bera_balance":      "Average match to population mix  (higher = fairer)",
    "violation_rate":        "Share of unfair clusters  (lower = better)",
    "tree_fidelity":         "Share of points explained by simple rules  (higher = better)",
    "tree_depth":            "Rule depth  (lower = simpler)",
    "tree_leaves":           "Number of rules  (lower = simpler)",
    "pam_cost":              "Average distance to cluster representative  (lower = tighter)",
    "cost_of_fairness":      "Quality loss from fairness  (1.0 = none, >1 = loss)",
    "avg_exemplar_coverage": "Average representative coverage  (higher = better)",
    "min_exemplar_coverage": "Worst representative coverage  (higher = better)",
    "avg_medoid_distance":   "Average distance to representative  (lower = tighter)",
    "runtime_s":             "Runtime in seconds  (lower = faster)",
}

# Static threshold reference lines (metric → (value, legend label))
# Dynamic thresholds (e.g. min_balance p/q) are passed into plot_metric_comparison
# via the `thresholds` argument from visualize_dataset().
METRIC_THRESHOLDS = {
    "violation_rate":   (0.0,  "Ideal: zero violations"),
    "cost_of_fairness": (1.0,  "Baseline: no fairness cost"),
    "tree_fidelity":    (0.90, "90% fidelity target"),
}

# Metrics shown in bar charts (key published metrics only — skip internal/legacy)
DEFAULT_METRICS = [
    "silhouette", "davies_bouldin", "sse",
    "min_balance", "min_bera_balance", "violation_rate",
    # "avg_dp_gap",  # commented out
    "tree_fidelity", "pam_cost", "cost_of_fairness",
]

# Friendly algorithm display names
ALGO_LABELS = {
    "kmeans_baseline":   "K-Means",
    "kmedians_baseline": "K-Medians",
    "fair_kmedians":     "Fair K-Medians\n(Fairlets)",
    "bounded_rep":       "Bounded-Rep\n(Bera et al.)",
    "fair_kmedoids":     "Fair K-Medoids (Fairlets)",
    "kmedoids_baseline": "K-Medoids",
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _pretty_algo(name):
    return ALGO_LABELS.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric comparison across clustering algorithms
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(df_results, dataset_name, output_dir,
                           metrics=None, thresholds=None):
    """
    Save one bar chart per metric comparing all clustering algorithms.

    Parameters
    ----------
    df_results   : pd.DataFrame  (one row per algorithm, 'model' column required)
    dataset_name : str
    output_dir   : str
    metrics      : list[str] or None  — defaults to DEFAULT_METRICS ∩ df columns
    thresholds   : dict {metric: (value, label)} or None
        Extra per-metric threshold lines (e.g. {'min_balance': (0.5, 'p/q = 1/2')}).
        Merged with METRIC_THRESHOLDS; caller values take priority.

    Returns
    -------
    list[str] : paths of saved figures
    """
    _ensure_dir(output_dir)

    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df_results.columns]

    # Merge static + dynamic thresholds
    effective_thresholds = dict(METRIC_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    saved = []
    algos        = df_results["model"].tolist()
    algo_display = [_pretty_algo(a) for a in algos]

    # Professional muted palette (colorblind-friendly)
    PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(algos))]

    for metric in metrics:
        if metric not in df_results.columns:
            continue

        values = df_results[metric].values

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(algo_display, values, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.6)

        # Value annotations on top of each bar
        for bar, v in zip(bars, values):
            if pd.isna(v):
                continue
            label = f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.005,
                label,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        # Threshold reference line
        if metric in effective_thresholds:
            tval, tlabel = effective_thresholds[metric]
            ax.axhline(tval, color="#CC0000", linestyle="--",
                       linewidth=1.4, alpha=0.85,
                       label=f"── {tlabel}  ({tval})")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

        # Titles and axis labels
        title_text = METRIC_LABELS.get(metric, metric)
        ax.set_title(
            f"{title_text}\n"
            f"Dataset: {dataset_name.upper()}",
            fontsize=10, pad=10,
        )
        ax.set_ylabel(
            METRIC_YLABELS.get(metric, metric),
            fontsize=9,
        )
        ax.set_xlabel("Clustering algorithm", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(
            output_dir, f"metric_{metric}_{dataset_name}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top-5 feature importance per cluster
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance_per_cluster(X, labels, feature_names,
                                           top_n=5, random_state=42,
                                           sensitive_col_prefix="sensitive"):
    """
    For each cluster k, train a one-vs-rest Random Forest classifier
    (cluster k vs the rest) and return the top-N most important features.

    Sensitive-attribute columns are excluded so the chart cannot suggest
    that protected attributes 'define' a cluster.

    Returns
    -------
    dict
        {cluster_id : [(feature_name, importance), ...]}  length top_n each.
    """
    feature_names = list(feature_names)

    # Drop sensitive columns
    keep_idx = [i for i, n in enumerate(feature_names)
                if not n.lower().startswith(sensitive_col_prefix.lower())]
    X_use = X[:, keep_idx]
    names_use = [feature_names[i] for i in keep_idx]

    importance_per_cluster = {}
    for k in np.unique(labels):
        y_bin = (labels == k).astype(int)
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=None,
            n_jobs=-1, random_state=random_state,
        )
        clf.fit(X_use, y_bin)
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1][:top_n]
        importance_per_cluster[int(k)] = [
            (names_use[i], float(importances[i])) for i in order
        ]
    return importance_per_cluster


def plot_feature_importance_per_cluster(X, labels, feature_names,
                                        algorithm_name, dataset_name,
                                        output_dir, top_n=5,
                                        random_state=42):
    """
    Save ONE figure for `algorithm_name` showing the top-N features per
    cluster as a grid of horizontal bar charts.

    File name : feature_importance_<algorithm>_<dataset>.png
    """
    _ensure_dir(output_dir)

    importance = compute_feature_importance_per_cluster(
        X, labels, feature_names, top_n=top_n, random_state=random_state
    )

    n_clusters = len(importance)
    ncols = min(3, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for ax_idx, (cluster_id, feats) in enumerate(importance.items()):
        r, c = ax_idx // ncols, ax_idx % ncols
        ax = axes[r][c]

        names = [_short_name(f) for f, _ in feats][::-1]
        vals  = [v for _, v in feats][::-1]

        ax.barh(names, vals,
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(vals))),
                edgecolor="black", linewidth=0.5)
        ax.set_title(f"Cluster {cluster_id}", fontsize=11)
        ax.set_xlabel("Importance")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    # Hide any leftover empty axes
    for j in range(len(importance), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Top-{top_n} feature importance per cluster\n"
        f"Algorithm: {_pretty_algo(algorithm_name)}  |  Dataset: {dataset_name}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"feature_importance_{algorithm_name}_{dataset_name}.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _short_name(name, max_len=28):
    """Trim long one-hot feature names so they fit on the y-axis."""
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA scatter plot coloured by cluster label
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_scatter(X, labels, algorithm_name, dataset_name,
                         output_dir, sample_size=4000, random_state=42):
    """
    Save a 2-D PCA scatter plot for ONE algorithm, with each point coloured
    by its cluster label and a per-cluster centroid annotation.

    Large datasets are sub-sampled to `sample_size` for plotting clarity.
    File name : scatter_<algorithm>_<dataset>.png
    """
    _ensure_dir(output_dir)

    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        X_plot = X[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X
        labels_plot = labels

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_plot)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = np.unique(labels_plot)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for color, k in zip(colors, unique_labels):
        mask = labels_plot == k
        n_cluster = int(mask.sum())
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=14, alpha=0.65, color=color,
                   label=f"Cluster {k}  (n={n_cluster:,})", edgecolor="none")

        # Annotate cluster centre
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, marker="X", s=180,
                   color=color, edgecolor="black", linewidth=1.2)
        ax.text(cx, cy, f"  C{k}", fontsize=11,
                fontweight="bold", color="black")

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance explained)", fontsize=10)
    ax.set_title(
        f"Cluster assignment — PCA 2-D projection\n"
        f"Algorithm: {_pretty_algo(algorithm_name)}  |  Dataset: {dataset_name.upper()}\n"
        f"PCA chosen: linear, deterministic, preserves global variance structure "
        f"(total {(var_explained[0]+var_explained[1])*100:.1f}% retained)",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(
        output_dir, f"scatter_{algorithm_name}_{dataset_name}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Surrogate decision tree visual figure
# ─────────────────────────────────────────────────────────────────────────────

def _clean_feature_name(name, max_len=40):
    """Strip sklearn ColumnTransformer prefixes (num__, cat__) and truncate."""
    name = str(name)
    for prefix in ("num__", "cat__", "remainder__"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    return name[:max_len] if len(name) > max_len else name


def plot_decision_tree(X, labels, feature_names, algorithm_name,
                       dataset_name, output_dir,
                       max_depth=None, random_state=None,
                       sensitive_col_prefix="sensitive",
                       print_ascii=True):
    """
    Train a surrogate decision tree and save a clean, readable PNG.

    Design choices for readability (professor requirement):
    - White node backgrounds with light grey class tinting (no garish colours)
    - Gini impurity hidden to reduce per-node text clutter
    - Proportions shown instead of raw counts (shorter text per node)
    - Figure width scales with 2^depth so leaf nodes never overlap
    - Sensitive-attribute columns excluded from features (fairness)

    Reference: Moshkovitz et al. (2020) ICML — Explainable k-Means / k-Medians.
    """
    from src.config import TREE_MAX_DEPTH

    if max_depth    is None: max_depth    = TREE_MAX_DEPTH
    if random_state is None: random_state = RANDOM_STATE

    _ensure_dir(output_dir)

    feature_names = list(feature_names)

    # Drop sensitive columns (same exclusion as explainability_metrics)
    keep_idx = [i for i, n in enumerate(feature_names)
                if not n.lower().startswith(sensitive_col_prefix.lower())]
    X_use       = X[:, keep_idx]
    names_clean = [_clean_feature_name(feature_names[i]) for i in keep_idx]

    unique_labels = np.unique(labels)
    n_classes     = len(unique_labels)
    class_names   = [f"Cluster {k}" for k in unique_labels]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_use, labels)
    fidelity = clf.score(X_use, labels)

    depth   = clf.get_depth()
    n_nodes = clf.tree_.node_count
    n_leaves = clf.get_n_leaves()

    # Figure size: width = leaf_count * per_leaf inches (real leaf count, not 2^depth)
    # This prevents both over-large empty figures AND leaf overlap.
    leaf_width   = 3.5          # inches per actual leaf (was 3.2 / 2^depth heuristic)
    node_height  = 2.8          # inches per depth level (was 2.4)
    fig_w = max(22, n_leaves * leaf_width)
    fig_h = max(11, (depth + 1) * node_height)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Muted class colour map: soft blues/greens/oranges — professional palette
    # sklearn plot_tree uses these via the tree_.value array when filled=True
    # We keep filled=True but override after plotting with muted colours.
    _MUTED = ["#AEC6CF", "#C8E6C9", "#FFCCBC", "#E1BEE7",
              "#FFF9C4", "#B3E5FC", "#D7CCC8", "#CFD8DC"]
    class_colours = {k: _MUTED[i % len(_MUTED)]
                     for i, k in enumerate(unique_labels)}

    plot_tree(
        clf,
        feature_names=names_clean,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,     # hide Gini — reduces per-node clutter
        proportion=True,    # show proportions not raw counts — shorter text
        ax=ax,
        fontsize=10,        # fixed readable font (was scaled, caused tiny text)
        precision=2,
    )
    ax.margins(x=0.02, y=0.05)  # extra breathing room so leaf boxes do not clip

    # Post-render: recolour node patches to muted palette
    # sklearn fills nodes; we darken leaf edges and lighten interior nodes
    for collection in ax.collections:
        # patch face colours encode class via sklearn's internal rgb mapping
        # we remap to our muted palette by matching majority class per node
        pass  # leave sklearn fill as-is — it is already muted at proportion=True

    # Lighten all node patches to 60% opacity so text stays readable
    for patch in ax.patches:
        patch.set_alpha(0.55)

    ax.set_title(
        f"Cluster Explanation — Decision Tree Rules\n"
        f"Algorithm: {_pretty_algo(algorithm_name)}  |  "
        f"Dataset: {dataset_name.upper()}  |  "
        f"Depth: {depth}  |  Rules (leaves): {n_leaves}  |  "
        f"Explanation accuracy: {fidelity:.1%}",
        fontsize=10, pad=14,
    )

    # Add a concise legend explaining how to read the tree
    legend_text = (
        "How to read: each box shows the decision rule, the dominant cluster,\n"
        "and the proportion of points reaching that node. "
        "Shading intensity indicates cluster purity."
    )
    fig.text(0.5, -0.01, legend_text, ha="center", fontsize=8,
             style="italic", color="#444444")

    out_path = os.path.join(
        output_dir, f"decision_tree_{algorithm_name}_{dataset_name}.png"
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight")

    # Also save SVG — vector format, no overlap at any zoom level
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    # Companion: plain-English rule list (.txt) — easiest to drop into thesis text
    from sklearn.tree import export_text
    rules_text = export_text(
        clf, feature_names=names_clean, show_weights=True,
        max_depth=depth, decimals=2,
    )
    rules_path = out_path.replace(".png", "_rules.txt")
    header = (
        f"Decision-Tree Rules — {algorithm_name} on {dataset_name}\n"
        f"Depth: {depth} | Rules (leaves): {n_leaves} | "
        f"Explanation accuracy: {fidelity:.1%}\n"
        f"{'-' * 70}\n"
    )
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(header + rules_text)

    # ── ASCII tree to terminal — traditional `|---` indented rule view ───────
    if print_ascii:
        print()
        print("=" * 78)
        print(f"  Decision Tree — {algorithm_name} on {dataset_name}")
        print(f"  Depth: {depth} | Leaves: {n_leaves} | "
              f"Accuracy: {fidelity:.1%}")
        print("=" * 78)
        print(rules_text)
        print("=" * 78)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pareto front analysis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pareto_front(costs):
    """
    Return boolean mask of Pareto-optimal rows.
    All columns in `costs` are treated as minimize-objectives.
    To maximise a metric, pass its negated values.

    Parameters
    ----------
    costs : ndarray (n_algorithms, n_objectives)

    Returns
    -------
    ndarray (n_algorithms,) dtype=bool — True = Pareto-optimal
    """
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # row i is dominated if some other row is <= in all dims and < in at least one
        dominated_by_i = (
            np.all(costs <= costs[i], axis=1) &
            np.any(costs <  costs[i], axis=1)
        )
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False
    return is_pareto


def plot_pareto_front(df_results, dataset_name, output_dir):
    """
    Plot two 2-D Pareto trade-off diagrams per dataset:
      Panel A — Cluster Quality  vs. Fairness
               x: Sum of Squared Errors (minimize) — cluster tightness
               y: Minimum cluster balance (maximize) — fairness
      Panel B — Fairness  vs. Explainability
               x: Fairness violation rate (minimize)
               y: Surrogate tree fidelity (maximize) — explanation accuracy

    Pareto-optimal algorithms are highlighted with a star marker and
    connected by the Pareto frontier line.

    Parameters
    ----------
    df_results   : pd.DataFrame (one row per algorithm, 'model' column)
    dataset_name : str
    output_dir   : str

    Returns
    -------
    str : path of the saved figure
    """
    _ensure_dir(output_dir)

    required_A = {"sse", "min_balance"}
    required_B = {"violation_rate", "tree_fidelity"}
    has_A = required_A.issubset(df_results.columns)
    has_B = required_B.issubset(df_results.columns)

    if not (has_A or has_B):
        return None  # not enough metrics to plot

    n_panels = int(has_A) + int(has_B)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
    algos        = df_results["model"].tolist()
    algo_display = [_pretty_algo(a) for a in algos]
    colors       = [PALETTE[i % len(PALETTE)] for i in range(len(algos))]

    panel_idx = 0

    # ── Panel A: Quality vs Fairness ─────────────────────────────────────────
    if has_A:
        ax = axes[panel_idx]; panel_idx += 1

        x_vals = df_results["sse"].values.astype(float)
        y_vals = df_results["min_balance"].values.astype(float)

        # Pareto: minimise SSE, maximise min_balance → minimise −min_balance
        costs = np.column_stack([x_vals, -y_vals])
        valid = ~(np.isnan(costs).any(axis=1))
        pareto_mask = np.zeros(len(costs), dtype=bool)
        if valid.sum() >= 2:
            pareto_mask[valid] = _compute_pareto_front(costs[valid])

        for i, (xv, yv, disp, col) in enumerate(
                zip(x_vals, y_vals, algo_display, colors)):
            marker  = "*" if pareto_mask[i] else "o"
            ms      = 220 if pareto_mask[i] else 100
            ax.scatter(xv, yv, color=col, s=ms, marker=marker,
                       edgecolor="black", linewidth=0.8, zorder=3,
                       label=disp)
            ax.annotate(
                disp.replace("\n", " "),
                (xv, yv), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=col,
            )

        # Draw Pareto frontier
        pf_x = x_vals[pareto_mask]
        pf_y = y_vals[pareto_mask]
        if len(pf_x) >= 2:
            order = np.argsort(pf_x)
            ax.step(pf_x[order], pf_y[order], where="post",
                    color="#CC0000", linestyle="--", linewidth=1.2,
                    alpha=0.75, label="Pareto frontier", zorder=2)

        ax.set_xlabel(
            "Cluster spread  (lower = tighter clusters)",
            fontsize=9,
        )
        ax.set_ylabel(
            "Group balance in worst cluster  (higher = fairer)",
            fontsize=9,
        )
        ax.set_title(
            "Cluster Quality vs Fairness",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel B: Fairness vs Explainability ──────────────────────────────────
    if has_B:
        ax = axes[panel_idx]; panel_idx += 1

        x_vals = df_results["violation_rate"].values.astype(float)
        y_vals = df_results["tree_fidelity"].values.astype(float)

        # Pareto: minimise violation_rate, maximise tree_fidelity
        costs = np.column_stack([x_vals, -y_vals])
        valid = ~(np.isnan(costs).any(axis=1))
        pareto_mask = np.zeros(len(costs), dtype=bool)
        if valid.sum() >= 2:
            pareto_mask[valid] = _compute_pareto_front(costs[valid])

        for i, (xv, yv, disp, col) in enumerate(
                zip(x_vals, y_vals, algo_display, colors)):
            marker = "*" if pareto_mask[i] else "o"
            ms     = 220 if pareto_mask[i] else 100
            ax.scatter(xv, yv, color=col, s=ms, marker=marker,
                       edgecolor="black", linewidth=0.8, zorder=3,
                       label=disp)
            ax.annotate(
                disp.replace("\n", " "),
                (xv, yv), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color=col,
            )

        pf_x = x_vals[pareto_mask]
        pf_y = y_vals[pareto_mask]
        if len(pf_x) >= 2:
            order = np.argsort(pf_x)
            ax.step(pf_x[order], pf_y[order], where="post",
                    color="#CC0000", linestyle="--", linewidth=1.2,
                    alpha=0.75, label="Pareto frontier", zorder=2)

        ax.set_xlabel(
            "Share of unfair clusters  (lower = fairer)",
            fontsize=9,
        )
        ax.set_ylabel(
            "Share of points explained by simple rules  (higher = clearer)",
            fontsize=9,
        )
        ax.set_title(
            "Fairness vs Explainability",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Trade-off Analysis — Dataset: {dataset_name.upper()}\n"
        "★ = best balance (no other algorithm beats it on both axes)",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"pareto_{dataset_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end "run all algorithms then plot everything" helper
# ─────────────────────────────────────────────────────────────────────────────
#
# This re-uses the same building blocks the existing pipeline uses
# (load_dataset, compute_pq, find_optimal_k, run_kmeans, k_medians,
#  twagner_fairlet_decomposition, ...) but keeps the labels in memory so we
# can pass them to the plotting functions.  Nothing in src/pipeline.py is
# touched.

def _evaluate_for_plot(name, X, labels, sensitive, feature_names, p, q,
                       alpha=None, beta=None):
    """Collect all metrics for one algorithm result (mirrors pipeline.evaluate)."""
    row = {"model": name}
    row.update(clustering_metrics(X, labels))
    row.update(fairness_metrics(labels, sensitive, p, q, alpha=alpha, beta=beta))
    exp = explainability_metrics(X, labels, feature_names=feature_names)
    row["tree_fidelity"] = exp["tree_fidelity"]
    row["tree_depth"]    = exp["tree_depth"]
    row["tree_leaves"]   = exp["tree_leaves"]
    ex = exemplar_metrics(X, labels)
    row["pam_cost"]              = ex["pam_cost"]
    row["avg_exemplar_coverage"] = ex["avg_exemplar_coverage"]
    row["min_exemplar_coverage"] = ex["min_exemplar_coverage"]
    row["avg_medoid_distance"]   = ex["avg_medoid_distance"]
    return row


def visualize_dataset(dataset_name, output_dir=None, verbose=True):
    """Run all four clustering algorithms then save every chart family.

    Charts produced per algorithm:
      - PCA 2-D scatter plot (cluster membership)
      - Top-5 feature importance per cluster (one-vs-rest Random Forest)
      - Surrogate decision tree visual figure
    Plus one bar chart per metric across all algorithms.

    Parameters
    ----------
    dataset_name : str  -- one of 'bank', 'adult', 'compas', 'german'
    output_dir   : str or None  (defaults to results/figures/<dataset_name>/)
    verbose      : bool

    Returns
    -------
    df_results      : pd.DataFrame  (one row per algorithm, all metrics)
    labels_per_algo : dict {algorithm_name: ndarray of cluster labels}
    """
    if output_dir is None:
        output_dir = os.path.join(FIG_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Generating charts for: {dataset_name.upper()}")
        print(f"  Output folder        : {output_dir}")
        print(f"{'=' * 60}")

    # -- Load data + auto p/q + auto k (mirrors pipeline.run_dataset) ---------
    X, sensitive, feature_names = load_dataset(
        dataset_name, DATASETS[dataset_name], RANDOM_STATE
    )
    p, q = compute_pq(sensitive)

    k_override = DATASET_K_OVERRIDE.get(dataset_name)
    if k_override is not None:
        k = int(k_override)
    else:
        if verbose:
            print("  Detecting optimal k ...", end=" ", flush=True)
        k, _, _ = find_optimal_k(X, k_range=K_RANGE, random_state=RANDOM_STATE)
        if verbose:
            print(f"k = {k}")

    if verbose:
        print(f"  Samples = {X.shape[0]}  Features = {X.shape[1]}  "
              f"k = {k}  p/q = {p}/{q}")

    # ── Run all 4 algorithms ─────────────────────────────────────────────────
    labels_per_algo = {}
    metric_rows     = []

    if verbose: print("  [1/4] K-Means ...", end=" ", flush=True)
    t = time.time()
    labels_per_algo["kmeans_baseline"] = run_kmeans(X, k, RANDOM_STATE)
    metric_rows.append(_evaluate_for_plot(
        "kmeans_baseline", X, labels_per_algo["kmeans_baseline"],
        sensitive, feature_names, p, q,
    ))
    if verbose: print(f"done ({time.time()-t:.1f}s)")

    if verbose: print("  [2/4] K-Medians ...", end=" ", flush=True)
    t = time.time()
    labels_per_algo["kmedians_baseline"] = k_medians(X, k, random_state=RANDOM_STATE)
    metric_rows.append(_evaluate_for_plot(
        "kmedians_baseline", X, labels_per_algo["kmedians_baseline"],
        sensitive, feature_names, p, q,
    ))
    if verbose: print(f"done ({time.time()-t:.1f}s)")

    if verbose: print("  [3/4] Fair K-Medians (fairlets) ...", end=" ", flush=True)
    t = time.time()
    fairlets, fc_indices = twagner_fairlet_decomposition(X, sensitive, p, q)
    fc_labels = cluster_fairlet_centers(
        X[fc_indices], k, method="kmedians", random_state=RANDOM_STATE
    )
    labels_per_algo["fair_kmedians"] = assign_labels_from_fairlets(
        fairlets, fc_labels, X.shape[0]
    )
    metric_rows.append(_evaluate_for_plot(
        "fair_kmedians", X, labels_per_algo["fair_kmedians"],
        sensitive, feature_names, p, q,
    ))
    if verbose: print(f"done ({time.time()-t:.1f}s)")

    if verbose: print("  [4/4] Bounded-Rep (Bera et al.) ...", end=" ", flush=True)
    t = time.time()
    labels_per_algo["bounded_rep"] = bounded_representation_clustering(
        X, sensitive, k, random_state=RANDOM_STATE
    )
    _p_global = float(sensitive.mean())
    _alpha = max(0.0, _p_global - 0.15)
    _beta  = min(1.0, _p_global + 0.15)
    metric_rows.append(_evaluate_for_plot(
        "bounded_rep", X, labels_per_algo["bounded_rep"],
        sensitive, feature_names, p, q,
        alpha=_alpha, beta=_beta,
    ))
    if verbose: print(f"done ({time.time()-t:.1f}s)")

    # ── Cost of Fairness: PAM_cost_algo / PAM_cost_kmeans_baseline ──────────
    # Reference: Chierichetti et al. (2017) §4; Bera et al. (2019) Theorem 1.
    # PAM cost uses linear distances — same metric family as k-median objective
    # (OPT_fair / OPT_unconstrained). Value = 1.0 for baseline; > 1.0 = fairness
    # costs clustering quality.
    baseline_pam = next(
        (r["pam_cost"] for r in metric_rows if r["model"] == "kmeans_baseline"), None
    )
    for row in metric_rows:
        if baseline_pam and baseline_pam > 0:
            row["cost_of_fairness"] = row["pam_cost"] / baseline_pam
        else:
            row["cost_of_fairness"] = float("nan")

    df_results = pd.DataFrame(metric_rows)

    # ── Metric comparison bar charts ─────────────────────────────────────────
    # Pass p/q as a dynamic threshold for the min_balance chart
    pq_threshold = p / q
    dynamic_thresholds = {
        "min_balance": (pq_threshold, f"Required balance threshold  (p/q = {p}/{q} = {pq_threshold:.2f})"),
    }
    if verbose: print("  Saving metric-comparison charts ...", end=" ", flush=True)
    paths = plot_metric_comparison(df_results, dataset_name, output_dir,
                                   thresholds=dynamic_thresholds)
    if verbose: print(f"{len(paths)} charts")

    # ── Pareto front analysis ────────────────────────────────────────────────
    if verbose: print("  Saving Pareto front chart ...", end=" ", flush=True)
    pareto_path = plot_pareto_front(df_results, dataset_name, output_dir)
    if verbose and pareto_path:
        print(f"saved → {os.path.basename(pareto_path)}")
    elif verbose:
        print("skipped (insufficient metrics)")

    # ── Per-algorithm: scatter + feature importance + decision tree ──────────
    if verbose: print("  Saving scatter / feature-importance / decision-tree ...",
                      end=" ", flush=True)
    n_charts = 0
    for algo, algo_labels in labels_per_algo.items():
        plot_cluster_scatter(
            X, algo_labels,
            algorithm_name=algo, dataset_name=dataset_name,
            output_dir=output_dir,
        )
        plot_feature_importance_per_cluster(
            X, algo_labels, feature_names,
            algorithm_name=algo, dataset_name=dataset_name,
            output_dir=output_dir,
        )
        plot_decision_tree(
            X, algo_labels, feature_names,
            algorithm_name=algo, dataset_name=dataset_name,
            output_dir=output_dir,
        )
        n_charts += 3
    if verbose: print(f"done ({n_charts} charts, {len(labels_per_algo)} algorithms)")

    return df_results, labels_per_algo
