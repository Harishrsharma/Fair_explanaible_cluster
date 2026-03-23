# logger.py
# Persistent run log: every pipeline execution is appended to a CSV file.
#
# Log file: results/run_log.csv
# Each row = one model × one dataset × one run (with timestamp).
#
# This lets you track experiment history, compare runs across days,
# and reproduce any previous result by checking the logged k / p / q values.

import os
import csv
from datetime import datetime

LOG_COLS = [
    "timestamp", "dataset", "k", "p", "q", "model",
    "silhouette", "davies_bouldin", "inertia",
    "min_balance", "avg_balance", "violation_rate", "avg_dp_gap",
    "tree_fidelity", "tree_depth", "tree_leaves",
    "avg_exemplar_coverage", "runtime_s",
]


def log_run(results_df, dataset_name, k, p, q, log_dir="results"):
    """
    Append all rows from results_df (one row per model) to the run log CSV.

    Parameters
    ----------
    results_df  : pd.DataFrame  – output of run_dataset()
    dataset_name: str
    k           : int – number of clusters used
    p, q        : int – fairness balance parameters used
    log_dir     : str – directory where run_log.csv is written
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log.csv")
    file_exists = os.path.isfile(log_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        for _, row in results_df.iterrows():
            entry = {col: row.get(col, "") for col in LOG_COLS}
            entry["timestamp"]    = timestamp
            entry["dataset"]      = dataset_name
            entry["k"]            = k
            entry["p"]            = p
            entry["q"]            = q
            entry["model"]        = row.get("model", "")
            writer.writerow(entry)

    return log_path
