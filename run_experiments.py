"""
run_experiments.py - Main experiment runner for CrossSynth.

Runs the full experimental grid:
    For each dataset (Adult, Credit, Bank):
      For each partition_mode (random, correlated):
        For each K (3, 5):
          For each epsilon (1.0, 10.0):
            For each method (CrossSynth, Independent, Centralized, PrivBayes):
              For each seed (2 seeds):
                1. Split data into K parties
                2. Train generator with given epsilon budget
                3. Generate synthetic data (same size as original)
                4. Evaluate: ML utility, marginal distance, correlation error
                5. Save results

Usage:
    python run_experiments.py                          # Run all experiments
    python run_experiments.py --dataset adult           # Single dataset
    python run_experiments.py --method fedsynth         # Single method
    python run_experiments.py --partition random         # Only random partition
    python run_experiments.py --partition correlated     # Only correlated
    python run_experiments.py --seeds 3 --epochs 100    # More seeds/epochs
    python run_experiments.py --quick                   # Quick test run

Results are saved to results/ as CSV and LaTeX tables.
Two separate result tables are generated: one for random, one for correlated.
"""

import os
import sys
import io
import json
import time
import argparse
import warnings
import traceback
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

# Ensure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix encoding for Korean Windows (cp949) - force UTF-8 output
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datasets import get_dataset, partition_data
from synthesizer import get_generator
from metrics import (
    ml_utility,
    marginal_distance_1way,
    marginal_distance_2way,
    correlation_error,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_DATASETS = ["adult", "credit", "bank"]
ALL_METHODS = ["fedsynth", "independent", "centralized", "privbayes"]
ALL_K_VALUES = [3, 5]
ALL_EPSILONS = [1.0, 10.0]
ALL_PARTITIONS = ["random", "correlated"]

# Quick-test configuration (small grid for sanity check)
QUICK_DATASETS = ["credit"]
QUICK_METHODS = ["fedsynth", "independent", "privbayes"]
QUICK_K_VALUES = [3]
QUICK_EPSILONS = [1.0]
QUICK_PARTITIONS = ["random", "correlated"]


# ---------------------------------------------------------------------------
# Synthetic label generation
# ---------------------------------------------------------------------------

def generate_synthetic_labels(X_synthetic, X_real, y_real, cat_indices=None):
    """
    Generate synthetic labels by nearest-neighbor matching.

    For each synthetic sample, find the nearest real sample and copy its label.
    This is a simple but effective approach for classification tasks.

    Parameters
    ----------
    X_synthetic : np.ndarray, shape (n_syn, d)
    X_real : np.ndarray, shape (n_real, d)
    y_real : np.ndarray, shape (n_real,)
    cat_indices : list or None (unused, kept for API compatibility)

    Returns
    -------
    y_synthetic : np.ndarray, shape (n_syn,)
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(X_real)
    _, indices = nn.kneighbors(X_synthetic)
    y_synthetic = y_real[indices.flatten()]
    return y_synthetic


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(dataset_name, method_name, K, epsilon, seed,
                          partition_mode="random", epochs=50, verbose=True):
    """
    Run a single experiment configuration.

    Parameters
    ----------
    dataset_name : str
    method_name : str
    K : int - number of parties
    epsilon : float - privacy budget
    seed : int - random seed
    partition_mode : str - 'random' or 'correlated'
    epochs : int - training epochs for diffusion models
    verbose : bool

    Returns
    -------
    result : dict or None
        Dict with all metrics, or None if experiment failed.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = (f"[{dataset_name}|{method_name}|{partition_mode}|K={K}|eps={epsilon}|"
           f"seed={seed}]")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {tag}")
        print(f"{'='*70}")

    # Load and preprocess dataset
    ds = get_dataset(dataset_name, random_state=seed)
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]
    partition_col_idx = ds.get("partition_col_idx", 0)

    # Partition into K parties
    parties = partition_data(
        X_train, y_train, K=K, mode=partition_mode,
        random_state=seed, partition_col_idx=partition_col_idx
    )
    if verbose:
        print(f"  Partitioned into {K} parties (mode={partition_mode}):")
        for i, p in enumerate(parties):
            print(f"    Party {i}: {p['X'].shape[0]} samples")

    # Create generator
    gen_kwargs = {}
    if method_name in ["fedsynth", "independent", "centralized"]:
        gen_kwargs = {
            "hidden_dim": 256,
            "n_layers": 3,
            "n_timesteps": 1000,
        }
        if method_name == "fedsynth":
            gen_kwargs["cond_dim"] = 256
    elif method_name == "privbayes":
        gen_kwargs = {"n_bins": 15, "max_parents": 2}

    generator = get_generator(method_name, **gen_kwargs)

    # Check CTGAN availability
    if method_name == "ctgan" and not generator._available:
        print(f"  {tag} CTGAN not available, skipping.")
        return None

    # Train
    t_start = time.time()
    fit_kwargs = {"epsilon": epsilon}
    if method_name in ["fedsynth", "independent", "centralized"]:
        fit_kwargs["epochs"] = epochs
        fit_kwargs["batch_size"] = min(256, max(p["X"].shape[0] for p in parties))
        fit_kwargs["lr"] = 1e-3
        fit_kwargs["verbose"] = verbose
    elif method_name == "ctgan":
        fit_kwargs["epochs"] = min(epochs, 150)

    generator.fit(parties, **fit_kwargs)
    t_train = time.time() - t_start

    # Generate synthetic data
    n_samples = X_train.shape[0]
    t_start = time.time()
    X_synthetic = generator.generate(n_samples)
    t_gen = time.time() - t_start

    # Generate synthetic labels (by nearest-neighbor matching to real data)
    y_synthetic = generate_synthetic_labels(
        X_synthetic, X_train, y_train, ds.get("cat_indices")
    )

    # --- Evaluate ---

    # ML Utility: train on synthetic, test on real
    ml_res = ml_utility(
        X_synthetic, y_synthetic, X_test, y_test,
        task="classification"
    )

    # Marginal distances
    tv_1way, _ = marginal_distance_1way(X_synthetic, X_train, n_bins=20)
    tv_2way, _ = marginal_distance_2way(X_synthetic, X_train, n_bins=10)

    # Correlation error
    corr_err = correlation_error(X_synthetic, X_train)

    # Extract primary metrics
    catboost_f1 = ml_res.get("catboost", {}).get("f1", float("nan"))
    catboost_acc = ml_res.get("catboost", {}).get("accuracy", float("nan"))
    xgboost_f1 = ml_res.get("xgboost", {}).get("f1", float("nan"))
    xgboost_acc = ml_res.get("xgboost", {}).get("accuracy", float("nan"))

    result = {
        "dataset": dataset_name,
        "method": method_name,
        "partition": partition_mode,
        "K": K,
        "epsilon": epsilon,
        "seed": seed,
        "catboost_f1": catboost_f1,
        "catboost_acc": catboost_acc,
        "xgboost_f1": xgboost_f1,
        "xgboost_acc": xgboost_acc,
        "tv_1way": tv_1way,
        "tv_2way": tv_2way,
        "corr_error": corr_err,
        "n_train": n_samples,
        "n_synthetic": X_synthetic.shape[0],
        "time_train": t_train,
        "time_generate": t_gen,
        "privacy_epsilon": epsilon,
        "privacy_delta": 1e-5,
    }

    if verbose:
        print(f"\n  Results for {tag}:")
        print(f"    CatBoost F1={catboost_f1:.4f}  Acc={catboost_acc:.4f}")
        print(f"    XGBoost  F1={xgboost_f1:.4f}  Acc={xgboost_acc:.4f}")
        print(f"    TV 1-way={tv_1way:.4f}  TV 2-way={tv_2way:.4f}")
        print(f"    Corr Error={corr_err:.4f}")
        print(f"    Train time={t_train:.1f}s  Gen time={t_gen:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_summary_table(df_results, metric_col="catboost_f1"):
    """
    Format results into a summary table:
    rows = methods, columns = (dataset, K, epsilon).

    Parameters
    ----------
    df_results : pd.DataFrame
    metric_col : str

    Returns
    -------
    table_str : str (formatted ASCII table)
    """
    if df_results.empty:
        return "No results to display."

    # Average over seeds
    grouped = df_results.groupby(
        ["dataset", "method", "K", "epsilon"]
    )[metric_col].agg(["mean", "std"]).reset_index()

    # Pivot for display
    grouped["config"] = (
        grouped["dataset"] + " K=" + grouped["K"].astype(str)
        + " eps=" + grouped["epsilon"].astype(str)
    )
    grouped["value_str"] = grouped.apply(
        lambda r: (
            f"{r['mean']:.3f}"
            if (np.isnan(r["std"]) or r["std"] == 0)
            else f"{r['mean']:.3f}+-{r['std']:.3f}"
        ) if not np.isnan(r["mean"]) else "N/A",
        axis=1
    )

    pivot = grouped.pivot(
        index="method", columns="config", values="value_str"
    ).fillna("N/A")

    table = tabulate(pivot, headers="keys", tablefmt="grid",
                     showindex=True)
    return table


def generate_latex_table(df_results, metric_col="catboost_f1",
                         caption="ML Utility (CatBoost F1)",
                         partition_label=""):
    """
    Generate a LaTeX table from results.

    Parameters
    ----------
    df_results : pd.DataFrame
    metric_col : str
    caption : str
    partition_label : str
        Label appended to caption (e.g. "Random" or "Correlated").

    Returns
    -------
    latex_str : str
    """
    if df_results.empty:
        return "% No results"

    if partition_label:
        caption = f"{caption} -- {partition_label} Partitioning"

    # Average over seeds
    grouped = df_results.groupby(
        ["dataset", "method", "K", "epsilon"]
    )[metric_col].agg(["mean", "std"]).reset_index()

    # Build LaTeX
    configs = grouped.groupby(["dataset", "K", "epsilon"]).size().reset_index()
    methods = sorted(grouped["method"].unique())

    n_configs = len(configs)
    header = " & ".join(
        [f"{r['dataset']}(K={r['K']},eps={r['epsilon']})"
         for _, r in configs.iterrows()]
    )

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{l" + "c" * n_configs + "}")
    lines.append("\\hline")
    lines.append("Method & " + header + " \\\\")
    lines.append("\\hline")

    for method in methods:
        row_vals = [method.replace("_", "\\_")]
        for _, cfg in configs.iterrows():
            mask = (
                (grouped["dataset"] == cfg["dataset"])
                & (grouped["method"] == method)
                & (grouped["K"] == cfg["K"])
                & (grouped["epsilon"] == cfg["epsilon"])
            )
            sub = grouped[mask]
            if len(sub) > 0:
                m = sub["mean"].values[0]
                s = sub["std"].values[0]
                if not np.isnan(m):
                    row_vals.append(f"{m:.3f}$\\pm${s:.3f}")
                else:
                    row_vals.append("N/A")
            else:
                row_vals.append("--")
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CrossSynth Experiment Runner"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=None,
        help="Datasets to run (adult/credit/bank). Default: all."
    )
    parser.add_argument(
        "--method", type=str, nargs="+", default=None,
        help="Methods to run. Default: all."
    )
    parser.add_argument(
        "--K", type=int, nargs="+", default=None,
        help="Number of parties. Default: [3, 5]."
    )
    parser.add_argument(
        "--epsilon", type=float, nargs="+", default=None,
        help="Privacy budgets. Default: [1.0, 10.0]."
    )
    parser.add_argument(
        "--partition", type=str, nargs="+", default=None,
        help="Partition modes: random, correlated. Default: both."
    )
    parser.add_argument(
        "--seeds", type=int, default=2,
        help="Number of random seeds (default: 2)."
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs for diffusion models (default: 100)."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test run with minimal grid."
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results/)."
    )

    args = parser.parse_args()

    # Resolve output directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build experiment grid
    if args.quick:
        datasets = QUICK_DATASETS
        methods = QUICK_METHODS
        k_values = QUICK_K_VALUES
        epsilons = QUICK_EPSILONS
        partition_modes = QUICK_PARTITIONS
        n_seeds = 1
        epochs = min(args.epochs, 10)
        print("*** QUICK TEST MODE ***")
    else:
        datasets = args.dataset if args.dataset else ALL_DATASETS
        methods = args.method if args.method else ALL_METHODS
        k_values = args.K if args.K else ALL_K_VALUES
        epsilons = args.epsilon if args.epsilon else ALL_EPSILONS
        partition_modes = args.partition if args.partition else ALL_PARTITIONS
        n_seeds = args.seeds
        epochs = args.epochs

    seeds = list(range(42, 42 + n_seeds))

    # Count total experiments
    total = (len(datasets) * len(partition_modes) * len(k_values)
             * len(epsilons) * len(methods) * len(seeds))
    print(f"\nExperiment Grid:")
    print(f"  Datasets   : {datasets}")
    print(f"  Partitions : {partition_modes}")
    print(f"  Methods    : {methods}")
    print(f"  K values   : {k_values}")
    print(f"  Epsilons   : {epsilons}")
    print(f"  Seeds      : {seeds}")
    print(f"  Epochs     : {epochs}")
    print(f"  Total experiments: {total}")
    print(f"  Output dir: {output_dir}")
    print()

    # Run experiments
    all_results = []
    completed = 0
    failed = 0
    skipped = 0

    for dataset_name in datasets:
        for partition_mode in partition_modes:
            for K in k_values:
                for epsilon in epsilons:
                    for method_name in methods:
                        for seed in seeds:
                            completed += 1
                            progress = f"[{completed}/{total}]"

                            try:
                                print(f"\n{progress} Running: "
                                      f"{dataset_name}/{partition_mode}/"
                                      f"{method_name}/"
                                      f"K={K}/eps={epsilon}/seed={seed}")

                                result = run_single_experiment(
                                    dataset_name=dataset_name,
                                    method_name=method_name,
                                    K=K,
                                    epsilon=epsilon,
                                    seed=seed,
                                    partition_mode=partition_mode,
                                    epochs=epochs,
                                    verbose=args.verbose,
                                )

                                if result is not None:
                                    all_results.append(result)
                                    # Save incrementally
                                    df_tmp = pd.DataFrame(all_results)
                                    df_tmp.to_csv(
                                        os.path.join(output_dir,
                                                     "results_partial.csv"),
                                        index=False
                                    )
                                else:
                                    skipped += 1

                            except Exception as e:
                                failed += 1
                                print(f"\n  *** ERROR in {progress}: {e}")
                                traceback.print_exc()
                                all_results.append({
                                    "dataset": dataset_name,
                                    "method": method_name,
                                    "partition": partition_mode,
                                    "K": K,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                    "error": str(e),
                                })

    # ---------------------------------------------------------------------------
    # Save final results
    # ---------------------------------------------------------------------------

    if not all_results:
        print("\nNo results collected. Exiting.")
        return

    df_results = pd.DataFrame(all_results)

    # Remove error-only rows for metric analysis
    df_valid = df_results.dropna(subset=["catboost_f1"]) if "catboost_f1" in df_results.columns else pd.DataFrame()

    # Save CSV
    csv_path = os.path.join(output_dir, "results_all.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary tables -- separate for each partition mode
    for pm in partition_modes:
        df_pm = df_valid[df_valid["partition"] == pm] if not df_valid.empty else pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"  SUMMARY: CatBoost F1 -- {pm.upper()} Partitioning")
        print(f"  (Train on Synthetic, Test on Real)")
        print(f"{'='*80}")
        if not df_pm.empty:
            print(format_summary_table(df_pm, "catboost_f1"))
        else:
            print("  No results for this partition mode.")

        print(f"\n{'='*80}")
        print(f"  SUMMARY: 1-Way Marginal TV Distance -- {pm.upper()} "
              f"Partitioning (lower is better)")
        print(f"{'='*80}")
        if not df_pm.empty:
            print(format_summary_table(df_pm, "tv_1way"))

        print(f"\n{'='*80}")
        print(f"  SUMMARY: Correlation Error -- {pm.upper()} "
              f"Partitioning (lower is better)")
        print(f"{'='*80}")
        if not df_pm.empty:
            print(format_summary_table(df_pm, "corr_error"))

    # Generate and save LaTeX tables -- separate per partition mode
    if not df_valid.empty:
        for pm in partition_modes:
            df_pm = df_valid[df_valid["partition"] == pm]
            if df_pm.empty:
                continue

            pm_label = pm.capitalize()
            for metric_col, caption in [
                ("catboost_f1",
                 "ML Utility: CatBoost F1 (higher is better)"),
                ("xgboost_f1",
                 "ML Utility: XGBoost F1 (higher is better)"),
                ("tv_1way",
                 "Statistical Fidelity: 1-Way TV Distance (lower is better)"),
                ("tv_2way",
                 "Statistical Fidelity: 2-Way TV Distance (lower is better)"),
                ("corr_error",
                 "Cross-Party Correlation Error (lower is better)"),
            ]:
                if metric_col in df_pm.columns:
                    latex = generate_latex_table(
                        df_pm, metric_col, caption,
                        partition_label=pm_label
                    )
                    fname = f"table_{metric_col}_{pm}.tex"
                    fpath = os.path.join(output_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(latex)
                    print(f"  LaTeX table saved: {fpath}")

    # Save experiment config
    config = {
        "datasets": datasets,
        "methods": methods,
        "partition_modes": partition_modes,
        "k_values": k_values,
        "epsilons": epsilons,
        "seeds": seeds,
        "epochs": epochs,
        "total_experiments": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total: {total} | Succeeded: {total - failed - skipped} | "
          f"Failed: {failed} | Skipped: {skipped}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
