# main.py
# Entry point for the fair explainable clustering thesis pipeline.
#
# Usage:
#   python main.py                          # run all 4 datasets + charts
#   python main.py --dataset german         # run german dataset + auto charts
#   python main.py --dataset bank adult     # run subset + auto charts
#   python main.py --no-save                # skip saving metrics CSV to disk
#   python main.py --no-plots               # skip chart generation
#   python main.py --plots-only --dataset german  # charts only (no metrics CSV)
#
# Charts are saved to: results/figures/<dataset>/

import sys
import argparse

# Force UTF-8 output on Windows so special characters (→, ★, …) in pipeline
# print statements don't crash with UnicodeEncodeError on cp1252 terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import run_all_datasets, run_dataset
from src.config import DATASETS
from src.visualization import visualize_dataset, FIG_ROOT


def _print_summary(df):
    """
    Print the final all-datasets summary as a rich terminal table.
    Falls back to plain pandas output if rich is not installed.

    Colour conventions
    ------------------
    violation_rate = 0.000  → bold bright_green  (fully fair)
    violation_rate > 0.000  → red
    pareto_optimal = True   → gold star ★
    Best value per column   → bold bright_green + underline
    Each model              → distinct colour
    Each dataset            → distinct colour in first column
    """
    try:
        from rich.console import Console
        from rich.table   import Table
        from rich         import box as rbox
    except ImportError:
        # Plain fallback
        cols = [c for c in [
            "dataset", "model", "min_balance", "violation_rate",
            "silhouette", "davies_bouldin", "sse", "tree_fidelity",
            "avg_dp_gap", "cost_of_fairness", "pam_cost", "pareto_optimal",
        ] if c in df.columns]
        print("\nDone. Summary:")
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    COLS = [
        ("dataset",             "Dataset",        "left"),
        ("model",               "Model",          "left"),
        ("min_balance",         "Min Bal ↑",      "right"),
        ("violation_rate",      "Violation ↓",    "right"),
        ("silhouette",          "Silhouette ↑",   "right"),
        ("davies_bouldin",      "Davies-B ↓",     "right"),
        ("sse",                 "SSE ↓",          "right"),
        ("avg_dp_gap",          "DP Gap ↓",       "right"),
        ("cost_of_fairness",    "FairCost ↓",     "right"),
        ("tree_fidelity",       "Tree Fid ↑",     "right"),
        ("pam_cost",            "PAM Cost ↓",     "right"),
        ("pareto_optimal",      "Pareto",         "center"),
    ]
    BETTER_HIGH = {"silhouette", "min_balance", "tree_fidelity"}
    BETTER_LOW  = {"davies_bouldin", "violation_rate", "sse", "avg_dp_gap",
                   "pam_cost", "cost_of_fairness"}

    MODEL_STYLE = {
        "kmeans_baseline":   "bright_blue",
        "kmedians_baseline": "yellow",
        "fair_kmedians":     "bright_green",
        "bounded_rep":       "magenta",
    }
    DATASET_STYLE = {
        "bank":   "bold cyan",
        "adult":  "bold bright_magenta",
        "compas": "bold bright_yellow",
        "german": "bold white",
    }

    import pandas as pd
    present = [(col, hdr, just) for col, hdr, just in COLS if col in df.columns]

    # Best value per numeric column across ALL datasets combined
    best_val = {}
    for col, _, _ in present:
        if col in ("model", "dataset", "pareto_optimal"):
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        best_val[col] = vals.max() if col in BETTER_HIGH else vals.min()

    table = Table(
        title="[bold white]Final Summary — All Datasets[/bold white]",
        box=rbox.DOUBLE_EDGE,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=True,
    )
    for col, hdr, just in present:
        table.add_column(hdr, justify=just, no_wrap=True)

    prev_ds = None
    for _, row in df.iterrows():
        cells = []
        for col, _, _ in present:
            val = row.get(col, "")

            if col == "dataset":
                ds = str(val)
                if ds != prev_ds:
                    s = DATASET_STYLE.get(ds, "white")
                    cells.append(f"[{s}]{ds}[/{s}]")
                else:
                    cells.append("")

            elif col == "model":
                s = MODEL_STYLE.get(str(val), "white")
                cells.append(f"[{s}]{val}[/{s}]")

            elif col == "pareto_optimal":
                if val is True or str(val).lower() == "true":
                    cells.append("[bold yellow]★ Yes[/bold yellow]")
                else:
                    cells.append("[dim]·[/dim]")

            elif col == "violation_rate":
                try:
                    num  = float(val)
                    fval = f"{num:.4f}"
                    if num == 0.0:
                        cells.append(f"[bold bright_green]{fval}[/bold bright_green]")
                    else:
                        cells.append(f"[red]{fval}[/red]")
                except (TypeError, ValueError):
                    cells.append("N/A")

            else:
                try:
                    num  = float(val)
                    fval = f"{num:.4f}"
                    if col in best_val and abs(num - best_val[col]) < 1e-9:
                        cells.append(
                            f"[bold bright_green underline]{fval}"
                            f"[/bold bright_green underline]"
                        )
                    else:
                        cells.append(f"[white]{fval}[/white]")
                except (TypeError, ValueError):
                    cells.append(str(val))

        prev_ds = str(row.get("dataset", ""))
        table.add_row(*cells)

    console = Console()
    console.print()
    console.print(table)
    console.print()


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
        help="Do not save metrics CSV to disk."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip chart generation (metric comparison, feature importance, "
             "scatter plots). Charts are generated by default."
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip the metrics pipeline; only generate the supervisor charts."
    )
    args = parser.parse_args()

    save     = not args.no_save
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

    _print_summary(df)

    # ── Auto-generate charts (always, unless --no-plots is passed) ────────
    if not args.no_plots:
        print("\nGenerating charts ...")
        for ds in datasets:
            visualize_dataset(ds, verbose=True)
        print(f"\nAll figures saved under: {FIG_ROOT}/")


if __name__ == "__main__":
    main()
