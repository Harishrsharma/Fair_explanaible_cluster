# main.py
# Entry point for the fair explainable clustering thesis pipeline.
#
# Usage:
#   python main.py                      # run all 4 datasets
#   python main.py --dataset bank       # run one dataset
#   python main.py --dataset bank adult # run subset
#   python main.py --no-save            # skip saving results to disk

import argparse
from src.pipeline import run_all_datasets, run_dataset
from src.config import DATASETS


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
    args = parser.parse_args()

    save = not args.no_save

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


if __name__ == "__main__":
    main()
