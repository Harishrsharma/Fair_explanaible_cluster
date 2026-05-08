# run_visualizations.py
# Standalone entry point for generating supervisor-requested charts WITHOUT
# changing anything in the existing pipeline.
#
# This is now a thin CLI wrapper around src.visualization.visualize_dataset().
# The same function is also called from main.py when you pass --plots, so:
#
#   python main.py --dataset german --plots     # metrics CSV + figures
#   python run_visualizations.py --dataset german  # figures only
#
# Output layout (per dataset):
#   results/figures/<dataset>/metric_<metric>_<dataset>.png
#   results/figures/<dataset>/feature_importance_<algorithm>_<dataset>.png
#   results/figures/<dataset>/scatter_<algorithm>_<dataset>.png

import os
import argparse

from src.config import DATASETS
from src.visualization import visualize_dataset, FIG_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison/feature/scatter charts for the "
                    "fair explainable clustering pipeline."
    )
    parser.add_argument(
        "--dataset", nargs="*", default=None,
        choices=list(DATASETS.keys()),
        help="Dataset(s) to visualize. Defaults to all 4.",
    )
    args = parser.parse_args()

    datasets = args.dataset if args.dataset else list(DATASETS.keys())
    os.makedirs(FIG_ROOT, exist_ok=True)

    for ds in datasets:
        visualize_dataset(ds, verbose=True)

    print(f"\nAll figures saved under: {FIG_ROOT}")


if __name__ == "__main__":
    main()
