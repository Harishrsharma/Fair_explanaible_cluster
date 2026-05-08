# pipeline.py
# Main experiment orchestrator for the thesis:
#   "Fair Explainable Clustering: from Fairness to Explainability"
#   Author: Harish Sharma  |  Supervisors: Prof. Dr. F. Hopfgartner & Dr. T. Le Quy
#
# Active methods per dataset (K-Medoids commented out per scope decision):
#   Baselines  : (1) K-Means   (2) K-Medians
#   Fair       : (3) Twagner Fairlets + K-Medians
#
# Key features:
#   * Auto p/q       -- ceil(n_maj/n_min) ensures threshold is always achievable
#   * Auto k         -- elbow method on KMeans inertia    (find_optimal_k)
#   * silhouette_pca -- silhouette in PCA space (corrects high-dim bias)
#   * avg_balance    -- per-cluster average balance alongside min_balance
#   * sse            -- Sum of Squared Errors for all methods
#   * Exemplar       -- avg/min coverage, avg medoid distance
#   * Run logger     -- every run appended to results/run_log.csv
#
# References
# ----------
# Fairlets (Twagner quadtree decomposition):
#   Chierichetti et al. (2017) "Fair Clustering Through Fairlets." NeurIPS.
#   https://proceedings.neurips.cc/paper/2017/hash/978fce5bcc4501f762b2523a5f23b66c-Abstract.html
#   GitHub reference impl: https://github.com/MilkaLichtblau/Multinomial_Fairlets
#
# K-Medians (L1-norm clustering baseline, professor-requested):
#   Bradley et al. (1997) "Clustering via Concave Minimization." NeurIPS.
#
# Explainability (surrogate decision tree):
#   Moshkovitz et al. (2020) "Explainable k-Means and k-Medians Clustering."
#   ICML 2020. https://arxiv.org/abs/2002.12538
#
# Elbow method for k selection:
#   Thorndike (1953) "Who Belongs in the Family?" Psychometrika.

import os
import time
import numpy as np
import pandas as pd

from src.config import (
    RANDOM_STATE, DATASETS,
    SENSITIVE_CONFIGS, RESULTS_DIR,
    K_RANGE, DATASET_K_OVERRIDE,
)
from src.data_processing import load_dataset
from src.clustering import run_kmeans, kmedoids, k_medians, cluster_fairlet_centers, find_optimal_k
from src.fairness import (
    twagner_fairlet_decomposition,
    assign_labels_from_fairlets,
    fairness_metrics,
    compute_pq,
    bounded_representation_clustering,
)
from src.metrics import clustering_metrics
from src.explainability import explainability_metrics, exemplar_metrics
from src.logger import log_run


# -----------------------------------------------------------------------------
# Single-model evaluation
# -----------------------------------------------------------------------------

def evaluate(name, X, labels, sensitive, feature_names, p, q):
    """
    Evaluate one clustering result across all metric categories.
    Returns a flat dict ready to be added to a results DataFrame.
    """
    row = {"model": name}

    # 1. Clustering quality (silhouette, silhouette_pca, davies_bouldin, sse)
    row.update(clustering_metrics(X, labels))

    # 2. Fairness (min_balance, avg_balance, violation_rate, avg_dp_gap)
    row.update(fairness_metrics(labels, sensitive, p, q))

    # 3. Rule-based explainability (surrogate decision tree)
    exp = explainability_metrics(X, labels, feature_names=feature_names)
    row["tree_fidelity"]   = exp["tree_fidelity"]
    row["tree_depth"]      = exp["tree_depth"]
    row["tree_leaves"]     = exp["tree_leaves"]
    row["_decision_rules"] = exp["decision_rules"]   # stored separately

    # 4. Exemplar-based explainability (medoid prototypes + PAM cost)
    ex = exemplar_metrics(X, labels)
    row["pam_cost"]              = ex["pam_cost"]              # published metric
    row["avg_exemplar_coverage"] = ex["avg_exemplar_coverage"] # CSV only
    row["min_exemplar_coverage"] = ex["min_exemplar_coverage"] # CSV only
    row["avg_medoid_distance"]   = ex["avg_medoid_distance"]   # legacy alias

    return row


# -----------------------------------------------------------------------------
# Run one complete dataset experiment
# -----------------------------------------------------------------------------

def run_dataset(dataset_name, verbose=True):
    """
    Load data, run all active methods, evaluate, and return a DataFrame.

    Parameters
    ----------
    dataset_name : str  -- one of 'bank', 'adult', 'compas', 'german'
    verbose      : bool -- print progress messages

    Returns
    -------
    df_results : pd.DataFrame  (one row per method, one column per metric)
    rules_dict : dict          (method_name -> decision rule string)
    """
    path = DATASETS[dataset_name]
    sens_desc = SENSITIVE_CONFIGS[dataset_name]["description"]

    # -- Load data ------------------------------------------------------------
    t0 = time.time()
    X, sensitive, feature_names = load_dataset(dataset_name, path, RANDOM_STATE)
    n, d = X.shape
    n1 = int(sensitive.sum())
    n0 = n - n1

    # -- Auto-select p/q from actual group ratio ------------------------------
    p, q = compute_pq(sensitive)

    # -- Auto-select k via elbow method ---------------------------------------
    k_override = DATASET_K_OVERRIDE.get(dataset_name)
    if k_override is not None:
        k = int(k_override)
        k_source = "override"
    else:
        if verbose:
            print(f"\n  Running elbow method for k in {K_RANGE} ...", end=" ", flush=True)
        k, _, _ = find_optimal_k(X, k_range=K_RANGE, random_state=RANDOM_STATE)
        k_source = "elbow"
        if verbose:
            print(f"optimal k = {k}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dataset   : {dataset_name.upper()}")
        print(f"  Sensitive : {sens_desc}")
        print(f"  Groups    : group-1 = {n1} ({100*n1/n:.1f}%)  |  "
              f"group-0 = {n0} ({100*n0/n:.1f}%)")
        print(f"  k = {k} ({k_source})  |  p/q = {p}/{q}  "
              f"(balance threshold = {p/q:.3f})")
        print(f"  Samples = {n}  |  Features = {d}")
        print(f"{'='*60}")

    results    = []
    rules_dict = {}

    # -- Baseline 1: K-Means --------------------------------------------------
    if verbose: print(f"  [1/4] K-Means baseline ...", end=" ", flush=True)
    t = time.time()
    labels = run_kmeans(X, k, RANDOM_STATE)
    row = evaluate("kmeans_baseline", X, labels, sensitive, feature_names, p, q)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["kmeans_baseline"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # -- Baseline 2: K-Medoids (COMMENTED OUT) --------------------------------
    # NOTE: K-Medoids removed from active pipeline to keep scope focused.
    # K-Medoids is O(n^2) per iteration and functionally covered by K-Medians.
    # Uncomment the block below to re-enable.
    #
    # if verbose: print(f"  [--] K-Medoids baseline ...", end=" ", flush=True)
    # t = time.time()
    # labels = kmedoids(X, k, random_state=RANDOM_STATE)
    # row = evaluate("kmedoids_baseline", X, labels, sensitive, feature_names, p, q)
    # row["runtime_s"] = round(time.time() - t, 2)
    # rules_dict["kmedoids_baseline"] = row.pop("_decision_rules")
    # results.append(row)
    # if verbose: print(f"done ({row['runtime_s']}s)")

    # -- Baseline 2: K-Medians ------------------------------------------------
    if verbose: print(f"  [2/4] K-Medians baseline ...", end=" ", flush=True)
    t = time.time()
    labels = k_medians(X, k, random_state=RANDOM_STATE)
    row = evaluate("kmedians_baseline", X, labels, sensitive, feature_names, p, q)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["kmedians_baseline"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # -- Fairlet decomposition (shared for both fair methods) -----------------
    # Reference: Chierichetti et al. (2017) NeurIPS.
    # GitHub:    https://github.com/MilkaLichtblau/Multinomial_Fairlets
    if verbose: print(f"  [3/4] Fairlet decomposition ...", end=" ", flush=True)
    t_fl = time.time()
    fairlets, fc_indices = twagner_fairlet_decomposition(X, sensitive, p, q)
    fc_X = X[fc_indices]
    if verbose: print(f"done ({round(time.time()-t_fl, 2)}s)")

    # -- Fair K-Medoids (COMMENTED OUT) ---------------------------------------
    # NOTE: Kept commented out to match baseline scope (only K-Medians active).
    # Uncomment to re-enable Fair K-Medoids comparisons.
    #
    # if verbose: print(f"  [--] Fair K-Medoids ...", end=" ", flush=True)
    # t = time.time()
    # fc_labels = cluster_fairlet_centers(fc_X, k, method="kmedoids",
    #                                     random_state=RANDOM_STATE)
    # labels = assign_labels_from_fairlets(fairlets, fc_labels, n)
    # row = evaluate("fair_kmedoids", X, labels, sensitive, feature_names, p, q)
    # row["runtime_s"] = round(time.time() - t, 2)
    # rules_dict["fair_kmedoids"] = row.pop("_decision_rules")
    # results.append(row)
    # if verbose: print(f"done ({row['runtime_s']}s)")

    # -- Fair K-Medians -------------------------------------------------------
    if verbose: print(f"  Fair K-Medians ...", end=" ", flush=True)
    t = time.time()
    fc_labels = cluster_fairlet_centers(fc_X, k, method="kmedians",
                                        random_state=RANDOM_STATE)
    labels = assign_labels_from_fairlets(fairlets, fc_labels, n)
    row = evaluate("fair_kmedians", X, labels, sensitive, feature_names, p, q)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["fair_kmedians"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # -- Bounded Representation (Bera et al., 2019) ---------------------------
    # Proportionally fair clustering: each cluster's group fraction is bounded
    # within [alpha, beta] = [max(0.05, p-0.15), min(0.95, p+0.15)].
    # Uses greedy constrained assignment (practical approximation of LP/flow).
    # Reference: Bera et al. (2019) "Fair Algorithms for Clustering." NeurIPS.
    if verbose: print(f"  [4/4] Bounded-Rep (Bera et al.) ...", end=" ", flush=True)
    t = time.time()
    labels = bounded_representation_clustering(X, sensitive, k,
                                               random_state=RANDOM_STATE)
    row = evaluate("bounded_rep", X, labels, sensitive, feature_names, p, q)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["bounded_rep"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    total_time = round(time.time() - t0, 1)
    if verbose:
        print(f"\n  Total time: {total_time}s")

    df = pd.DataFrame(results)
    df.insert(0, "dataset", dataset_name)

    # Reorder columns for readability
    metric_cols = [
        "model",
        # -- Clustering quality -----------------------------------------------
        "silhouette", "silhouette_pca", "davies_bouldin", "sse",
        # -- Fairness ---------------------------------------------------------
        "min_balance", "avg_balance", "violation_rate", "avg_dp_gap",
        # -- Explainability: rule-based ----------------------------------------
        "tree_fidelity", "tree_depth", "tree_leaves",
        # -- Explainability: exemplar-based ------------------------------------
        "pam_cost",                                          # published metric
        "avg_exemplar_coverage", "min_exemplar_coverage",   # CSV only
        "avg_medoid_distance",                               # legacy alias
        "runtime_s",
    ]
    present = [c for c in metric_cols if c in df.columns]
    df = df[["dataset"] + present]

    if verbose:
        _print_table(df, p, q, k)

    # -- Log run to CSV -------------------------------------------------------
    log_path = log_run(df, dataset_name, k, p, q, log_dir=RESULTS_DIR)
    if verbose:
        print(f"  Run logged -> {log_path}")

    return df, rules_dict


# -----------------------------------------------------------------------------
# Run all 4 datasets
# -----------------------------------------------------------------------------

def run_all_datasets(datasets=None, save_results=True, verbose=True):
    """
    Run the full pipeline on all (or a subset of) thesis datasets.

    Parameters
    ----------
    datasets     : list of str or None  (runs all 4 if None)
    save_results : bool
    verbose      : bool

    Returns
    -------
    df_all    : pd.DataFrame
    rules_all : dict  {dataset: {method: rules_string}}
    """
    if datasets is None:
        datasets = list(DATASETS.keys())

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_frames = []
    rules_all  = {}

    for ds in datasets:
        df, rules = run_dataset(ds, verbose=verbose)
        all_frames.append(df)
        rules_all[ds] = rules

        if save_results:
            out_csv = os.path.join(RESULTS_DIR, f"{ds}_results.csv")
            df.to_csv(out_csv, index=False)
            _save_rules(rules, os.path.join(RESULTS_DIR, f"{ds}_rules.txt"))
            if verbose:
                print(f"  Saved -> {out_csv}")

    df_all = pd.concat(all_frames, ignore_index=True)

    if save_results:
        combined_path = os.path.join(RESULTS_DIR, "all_datasets_results.csv")
        df_all.to_csv(combined_path, index=False)
        if verbose:
            print(f"\nCombined results saved -> {combined_path}")

    return df_all, rules_all


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _print_table(df, p, q, k):
    """Print a readable summary table to stdout."""
    display_cols = [
        "model",
        # Clustering quality
        "silhouette", "silhouette_pca", "davies_bouldin", "sse",
        # Fairness
        "min_balance", "avg_balance", "violation_rate", "avg_dp_gap",
        # Explainability
        "tree_fidelity",
        # Exemplar
        "avg_exemplar_coverage", "min_exemplar_coverage", "avg_medoid_distance",
    ]
    cols = [c for c in display_cols if c in df.columns]
    print()
    print(f"  Balance threshold (p/q = {p}/{q} = {p/q:.3f})  |  k = {k}")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


def _save_rules(rules_dict, path):
    """Write decision rules for all methods to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        for method, rules in rules_dict.items():
            f.write(f"{'='*60}\n")
            f.write(f"  Method: {method}\n")
            f.write(f"{'='*60}\n")
            f.write(rules)
            f.write("\n\n")
