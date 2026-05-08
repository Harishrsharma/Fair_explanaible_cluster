# main.py
# Entry point for the fair explainable clustering thesis pipeline.
#
# Usage:
#   python main.py                          # run all 4 datasets
#   python main.py --dataset bank           # run one dataset
#   python main.py --dataset bank adult     # run subset
#   python main.py --no-save                # skip saving results to disk
#   python main.py --dataset bank --plots   # also save comparison/feature/scatter charts
#   python main.py --plots-only --dataset bank  # only generate the charts, skip metrics CSV

import argparse
from src.pipeline import run_all_datasets, run_dataset
from src.config import DATASETS
from src.visualization import visualize_dataset, FIG_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Fair Explainable Clustering – Thesis Pipeline"
    )
    parser.add_argument(
        "--dataset", nargs="*", default=None,
        choices=list(DATASETS.keys()),
        help="Dataset(s) to run. Defaults to all 4."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save results to disk."
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Also generate comparison, feature-importance and scatter charts "
             f"under {FIG_ROOT}/<dataset>/."
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip the metrics pipeline; only generate the charts."
    )
    args = parser.parse_args()

    save = not args.no_save
    datasets = args.dataset if args.dataset else list(DATASETS.keys())

    # ── Plots-only fast path ──────────────────────────────────────────────
    if args.plots_only:
        for ds in datasets:
            visualize_dataset(ds, verbose=True)
        print(f"\nFigures saved under: {FIG_ROOT}/")
        return

    # ── Normal pipeline run ───────────────────────────────────────────────
    if args.dataset and len(args.dataset) == 1:
        df, rules = run_dataset(args.dataset[0], verbose=True)
    else:
        df, rules = run_all_datasets(
            datasets=args.dataset,
            save_results=save,
            verbose=True,
        )

    print("\nDone. Summary:")
    print(df[["dataset", "model", "min_balance", "violation_rate",
              "silhouette", "tree_fidelity"]].to_string(index=False))

    # ── Optional plots after the metrics run ──────────────────────────────
    if args.plots:
        print("\nGenerating charts ...")
        for ds in datasets:
            visualize_dataset(ds, verbose=True)
        print(f"\nFigures saved under: {FIG_ROOT}/")


if __name__ == "__main__":
    main()
