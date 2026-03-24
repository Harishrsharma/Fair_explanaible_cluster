# main.py

import pandas as pd
from src.config import *
from src.data_processing import load_bank_dataset
from src.clustering import run_kmeans, kmedoids
from src.fairness import fairlet_decomposition, fairness_metrics
from src.explainability import explainability_metrics
from src.metrics import clustering_metrics


def evaluate_model(name, X, labels, sensitive):

    results = {}

    cm = clustering_metrics(X, labels)
    fm = fairness_metrics(labels, sensitive, FAIR_P, FAIR_Q)
    em = explainability_metrics(
        X, labels,
        max_depth=TREE_MAX_DEPTH,
        random_state=RANDOM_STATE
    )

    results.update(cm)
    results.update(fm)
    results.update(em)

    results["model"] = name

    return results


def run_experiment(dataset_name):

    X, sensitive, _ = load_bank_dataset(
        DATASETS[dataset_name]
    )

    results = []

    # Vanilla
    vanilla_labels = run_kmeans(
        X, K_CLUSTERS, RANDOM_STATE
    )

    results.append(
        evaluate_model("vanilla_kmeans", X, vanilla_labels, sensitive)
    )

    # Fairlets
    fairlets = fairlet_decomposition(
        X, sensitive, FAIR_P, FAIR_Q
    )

    # Simplified clustering after fairlets
    fair_labels = run_kmeans(
        X, K_CLUSTERS, RANDOM_STATE
    )

    results.append(
        evaluate_model("fair_kmeans", X, fair_labels, sensitive)
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"results/{dataset_name}_results.csv", index=False)

    print(df_results)


if __name__ == "__main__":
    run_experiment("bank")
