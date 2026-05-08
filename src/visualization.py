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
# Display-friendly metric titles (used in chart titles / y-axis labels)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "silhouette":            "Silhouette score (higher = better)",
    "silhouette_pca":        "Silhouette in PCA space (higher = better)",
    "davies_bouldin":        "Davies-Bouldin index (lower = better)",
    "sse":                   "Sum of Squared Errors (lower = better)",
    "min_balance":           "Minimum cluster balance (higher = fairer)",
    "avg_balance":           "Average cluster balance (higher = fairer)",
    "violation_rate":        "Fairness violation rate (lower = fairer)",
    "avg_dp_gap":            "Average demographic parity gap (lower = fairer)",
    "tree_fidelity":         "Surrogate tree fidelity (higher = more explainable)",
    "tree_depth":            "Surrogate tree depth",
    "tree_leaves":           "Surrogate tree leaves",
    "pam_cost":              "PAM cost — (1/n)Σd(x,m_k) — (lower = tighter exemplars)",
    "cost_of_fairness":      "Cost of Fairness — SSE_fair / SSE_baseline (lower = cheaper fairness)",
    "avg_exemplar_coverage": "Average exemplar coverage (higher = better)",
    "min_exemplar_coverage": "Minimum exemplar coverage (higher = better)",
    "avg_medoid_distance":   "Average medoid distance (lower = tighter)",
    "runtime_s":             "Runtime (seconds)",
}

# Metrics shown in bar charts (key published metrics only — skip internal/legacy)
DEFAULT_METRICS = [
    "silhouette", "davies_bouldin", "sse",
    "min_balance", "violation_rate", "avg_dp_gap",
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
                           metrics=None):
    """
    Save one bar chart per metric, comparing every clustering algorithm.

    Parameters
    ----------
    df_results   : pd.DataFrame
        Output of pipeline.run_dataset (one row per algorithm).
        Must contain a 'model' column plus the metric columns.
    dataset_name : str
        Used in chart titles and the saved file name.
    output_dir   : str
        Directory in which to save figures.
    metrics      : list[str] or None
        Metrics to plot.  Defaults to every known metric present in df_results.

    Returns
    -------
    list[str] : paths of the saved figures
    """
    _ensure_dir(output_dir)

    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df_results.columns]

    saved = []
    algos = df_results["model"].tolist()
    algo_display = [_pretty_algo(a) for a in algos]
    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

    for metric in metrics:
        if metric not in df_results.columns:
            continue

        values = df_results[metric].values

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(algo_display, values, color=colors,
                      edgecolor="black", linewidth=0.6)

        # Annotate each bar with its value
        for bar, v in zip(bars, values):
            if pd.isna(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n"
                     f"Dataset: {dataset_name}", fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xlabel("Clustering algorithm")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        out_path = os.path.join(
            output_dir, f"metric_{metric}_{dataset_name}.png"
        )
        plt.savefig(out_path, dpi=150)
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
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=14, alpha=0.65, color=color,
                   label=f"Cluster {k}", edgecolor="none")

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
# 4. End-to-end "run all algorithms then plot everything" helper
# ─────────────────────────────────────────────────────────────────────────────
#
# This re-uses the same building blocks the existing pipeline uses
# (load_dataset, compute_pq, find_optimal_k, run_kmeans, k_medians,
#  twagner_fairlet_decomposition, ...) but keeps the labels in memory so we
# can pass them to the plotting functions.  Nothing in src/pipeline.py is
# touched.

def _evaluate_for_plot(name, X, labels, sensitive, feature_names, p, q):
    """Collect all metrics for one algorithm result (mirrors pipeline.evaluate)."""
    row = {"model": name}
    row.update(clustering_metrics(X, labels))
    row.update(fairness_metrics(labels, sensitive, p, q))
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
    metric_rows.append(_evaluate_for_plot(
        "bounded_rep", X, labels_per_algo["bounded_rep"],
        sensitive, feature_names, p, q,
    ))
    if verbose: print(f"done ({time.time()-t:.1f}s)")

    # ── Cost of Fairness: SSE_algo / SSE_kmeans_baseline ─────────────────────
    # Reference: Chierichetti et al. (2017) §4; Bera et al. (2019) Theorem 1.
    # Measures quality cost incurred to achieve fairness.
    # Value = 1.0 for baseline; > 1.0 means fairness costs SSE quality.
    baseline_sse = next(
        (r["sse"] for r in metric_rows if r["model"] == "kmeans_baseline"), None
    )
    for row in metric_rows:
        if baseline_sse and baseline_sse > 0:
            row["cost_of_fairness"] = row["sse"] / baseline_sse
        else:
            row["cost_of_fairness"] = float("nan")

    df_results = pd.DataFrame(metric_rows)

    # ── Metric comparison bar charts ─────────────────────────────────────────
    if verbose: print("  Saving metric-comparison charts ...", end=" ", flush=True)
    paths = plot_metric_comparison(df_results, dataset_name, output_dir)
    if verbose: print(f"{len(paths)} charts")

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
