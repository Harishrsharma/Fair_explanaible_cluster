# pipeline.py
# Main experiment orchestrator for the thesis:
#   "Fair Explainable Clustering: from Fairness to Explainability"
#   Author: Harish Sharma  |  Supervisors: Prof. Dr. F. Hopfgartner & Dr. T. Le Quy
#
# Pipeline runs 5 methods per dataset:
#   Baselines  : (1) K-Means   (2) K-Medoids   (3) K-Medians
#   Fair       : (4) Twagner Fairlets + K-Medoids
#                (5) Twagner Fairlets + K-Medians
#
# Reference paper for fairlets:
#   Chierichetti et al. (2017) "Fair Clustering Through Fairlets." NeurIPS.

import os
import time
import numpy as np
import pandas as pd

from src.config import (
    K_CLUSTERS, FAIR_P, FAIR_Q, RANDOM_STATE, DATASETS,
    SENSITIVE_CONFIGS, RESULTS_DIR
)
from src.data_processing import load_dataset
from src.clustering import run_kmeans, kmedoids, k_medians, cluster_fairlet_centers
from src.fairness import (
    twagner_fairlet_decomposition,
    assign_labels_from_fairlets,
    fairness_metrics,
)
from src.metrics import clustering_metrics
from src.explainability import explainability_metrics, exemplar_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, X, labels, sensitive, feature_names):
    """
    Evaluate one clustering result across all three metric categories.
    Returns a flat dict ready to be added to a results DataFrame.
    """
    row = {"model": name}

    # 1. Clustering quality
    row.update(clustering_metrics(X, labels))

    # 2. Fairness
    row.update(fairness_metrics(labels, sensitive, FAIR_P, FAIR_Q))

    # 3. Rule-based explainability (surrogate decision tree)
    exp = explainability_metrics(X, labels, feature_names=feature_names)
    row["tree_fidelity"] = exp["tree_fidelity"]
    row["tree_depth"]    = exp["tree_depth"]
    row["tree_leaves"]   = exp["tree_leaves"]
    # store rules separately (too long for a table row)
    row["_decision_rules"] = exp["decision_rules"]

    # 4. Exemplar-based explainability
    ex = exemplar_metrics(X, labels)
    row["avg_exemplar_coverage"] = ex["avg_exemplar_coverage"]

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Run one complete dataset experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name, verbose=True):
    """
    Load data, run all 5 methods, evaluate, and return a DataFrame of results.

    Parameters
    ----------
    dataset_name : str – one of 'bank', 'adult', 'compas', 'german'
    verbose      : bool – print progress messages

    Returns
    -------
    df_results : pd.DataFrame  (one row per method, one column per metric)
    rules_dict : dict          (method_name -> decision rule string)
    """
    path = DATASETS[dataset_name]
    sens_desc = SENSITIVE_CONFIGS[dataset_name]["description"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dataset : {dataset_name.upper()}")
        print(f"  Sensitive attribute : {sens_desc}")
        print(f"  k = {K_CLUSTERS}  |  p/q = {FAIR_P}/{FAIR_Q}")
        print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    t0 = time.time()
    X, sensitive, feature_names = load_dataset(dataset_name, path, RANDOM_STATE)
    n, d = X.shape
    n_minority = int(sensitive.sum())
    if verbose:
        print(f"  Loaded {n} samples, {d} features | "
              f"minority={n_minority} ({100*n_minority/n:.1f}%)")

    results = []
    rules_dict = {}

    # ── Baseline 1: K-Means ───────────────────────────────────────────────
    if verbose: print("  [1/5] K-Means baseline ...", end=" ", flush=True)
    t = time.time()
    labels = run_kmeans(X, K_CLUSTERS, RANDOM_STATE)
    row = evaluate("kmeans_baseline", X, labels, sensitive, feature_names)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["kmeans_baseline"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # ── Baseline 2: K-Medoids ─────────────────────────────────────────────
    if verbose: print("  [2/5] K-Medoids baseline ...", end=" ", flush=True)
    t = time.time()
    labels = kmedoids(X, K_CLUSTERS, random_state=RANDOM_STATE)
    row = evaluate("kmedoids_baseline", X, labels, sensitive, feature_names)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["kmedoids_baseline"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # ── Baseline 3: K-Medians ─────────────────────────────────────────────
    if verbose: print("  [3/5] K-Medians baseline ...", end=" ", flush=True)
    t = time.time()
    labels = k_medians(X, K_CLUSTERS, random_state=RANDOM_STATE)
    row = evaluate("kmedians_baseline", X, labels, sensitive, feature_names)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["kmedians_baseline"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # ── Fair method 4: Twagner Fairlets + K-Medoids ───────────────────────
    if verbose: print("  [4/5] Twagner Fairlets + K-Medoids ...", end=" ", flush=True)
    t = time.time()
    fairlets, fc_indices = twagner_fairlet_decomposition(X, sensitive, FAIR_P, FAIR_Q)
    fc_X = X[fc_indices]                                   # fairlet centre features
    fc_labels = cluster_fairlet_centers(fc_X, K_CLUSTERS, method="kmedoids",
                                        random_state=RANDOM_STATE)
    labels = assign_labels_from_fairlets(fairlets, fc_labels, n)
    row = evaluate("fair_kmedoids", X, labels, sensitive, feature_names)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["fair_kmedoids"] = row.pop("_decision_rules")
    results.append(row)
    if verbose: print(f"done ({row['runtime_s']}s)")

    # ── Fair method 5: Twagner Fairlets + K-Medians ───────────────────────
    if verbose: print("  [5/5] Twagner Fairlets + K-Medians ...", end=" ", flush=True)
    t = time.time()
    # Reuse fairlets from step 4 (same decomposition)
    fc_labels = cluster_fairlet_centers(fc_X, K_CLUSTERS, method="kmedians",
                                        random_state=RANDOM_STATE)
    labels = assign_labels_from_fairlets(fairlets, fc_labels, n)
    row = evaluate("fair_kmedians", X, labels, sensitive, feature_names)
    row["runtime_s"] = round(time.time() - t, 2)
    rules_dict["fair_kmedians"] = row.pop("_decision_rules")
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
        "silhouette", "davies_bouldin", "inertia",
        "min_balance", "avg_balance", "violation_rate", "avg_dp_gap",
        "tree_fidelity", "tree_depth", "tree_leaves", "avg_exemplar_coverage",
        "runtime_s",
    ]
    present = [c for c in metric_cols if c in df.columns]
    df = df[["dataset"] + present]

    if verbose:
        _print_table(df)

    return df, rules_dict


# ─────────────────────────────────────────────────────────────────────────────
# Run all 4 datasets
# ─────────────────────────────────────────────────────────────────────────────

def run_all_datasets(datasets=None, save_results=True, verbose=True):
    """
    Run the full pipeline on all (or a subset of) thesis datasets.

    Parameters
    ----------
    datasets     : list of str or None (runs all 4 if None)
    save_results : bool – save CSV + rules text to RESULTS_DIR
    verbose      : bool

    Returns
    -------
    df_all    : pd.DataFrame – combined results for all datasets
    rules_all : dict         – {dataset: {method: rules_string}}
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
                print(f"  Saved → {out_csv}")

    df_all = pd.concat(all_frames, ignore_index=True)

    if save_results:
        combined_path = os.path.join(RESULTS_DIR, "all_datasets_results.csv")
        df_all.to_csv(combined_path, index=False)
        if verbose:
            print(f"\nCombined results saved → {combined_path}")

    return df_all, rules_all


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(df):
    """Print a readable summary table to stdout."""
    display_cols = [
        "model", "silhouette", "davies_bouldin",
        "min_balance", "violation_rate", "avg_dp_gap",
        "tree_fidelity",
    ]
    cols = [c for c in display_cols if c in df.columns]
    print()
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
