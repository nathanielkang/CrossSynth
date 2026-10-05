"""Run the reviewer-requested CrossSynth ablations from measured executions.

The script writes one long-form CSV and one JSON summary. It never reads or
updates manuscript tables. Run the smoke profile first, inspect overall ranks,
then use ``--profile paper`` for manuscript evidence.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from run_experiments import run_single_experiment


def _configurations(suites):
    configs = []
    if "conditioning" in suites:
        for mode in ["marginal", "constant", "random", "none"]:
            configs.append({
                "suite": "conditioning",
                "variant": mode,
                "cond_mode": mode,
                "M": 50,
                "selection": "domain_size",
                "eps_marginal_frac": 0.1,
            })
    if "marginals" in suites:
        for M in [10, 25, 50, 100]:
            configs.append({
                "suite": "marginals",
                "variant": f"domain_size_M{M}",
                "cond_mode": "marginal",
                "M": M,
                "selection": "domain_size",
                "eps_marginal_frac": 0.1,
            })
        configs.append({
            "suite": "marginals",
            "variant": "random_M50",
            "cond_mode": "marginal",
            "M": 50,
            "selection": "random",
            "eps_marginal_frac": 0.1,
        })
    if "budget" in suites:
        for fraction in [0.05, 0.1, 0.2, 0.5, 0.8]:
            configs.append({
                "suite": "budget",
                "variant": f"marginal_{fraction:.2f}",
                "cond_mode": "marginal",
                "M": 50,
                "selection": "domain_size",
                "eps_marginal_frac": fraction,
            })
    return configs


def _summarize(frame):
    metrics = ["catboost_f1", "xgboost_f1", "tv_1way", "tv_2way",
               "corr_error", "conditional_f1"]
    available = [metric for metric in metrics if metric in frame.columns]
    grouped = frame.groupby(["suite", "variant"], dropna=False)[available]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", nargs="+",
        choices=["conditioning", "marginals", "budget"],
        default=["conditioning", "marginals", "budget"],
    )
    parser.add_argument("--profile", choices=["smoke", "paper"],
                        default="smoke")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset", default="credit")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--eps-marginal-frac", type=float, default=None)
    parser.add_argument(
        "--calibrate-output", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--calibration-include-2way", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--calibration-passes", type=int, default=5)
    parser.add_argument("--calibration-damping", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/revision_ablations")
    args = parser.parse_args()

    paper = args.profile == "paper"
    epochs = args.epochs if args.epochs is not None else (100 if paper else 1)
    seeds = list(range(42, 42 + args.seeds))
    configs = _configurations(args.suite)
    if args.eps_marginal_frac is not None:
        for config in configs:
            config["eps_marginal_frac"] = float(args.eps_marginal_frac)
    rows = []

    for config in configs:
        for seed in seeds:
            result = run_single_experiment(
                dataset_name=args.dataset,
                method_name="fedsynth",
                K=args.K,
                epsilon=1.0,
                seed=seed,
                partition_mode="correlated",
                epochs=epochs,
                verbose=False,
                cond_mode=config["cond_mode"],
                M=config["M"],
                selection=config["selection"],
                eps_marginal_frac=config["eps_marginal_frac"],
                round_counts=True,
                marginal_mode="count",
                secure_aggregation=True,
                private_training=paper,
                normalization="local",
                joint_target=True,
                calibrate_output=args.calibrate_output,
                calibration_oversample=2.0,
                calibration_passes=args.calibration_passes,
                calibration_include_2way=args.calibration_include_2way,
                calibration_damping=args.calibration_damping,
            )
            result["suite"] = config["suite"]
            result["variant"] = config["variant"]
            result["run_profile"] = args.profile
            result["paper_protocol"] = bool(paper)
            rows.append(result)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    frame = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "revision_ablations.csv")
    frame.to_csv(raw_path, index=False)
    summary = _summarize(frame)
    summary_path = os.path.join(output_dir, "revision_ablations_summary.csv")
    summary.to_csv(summary_path, index=False)

    payload = {
        "profile": args.profile,
        "paper_protocol": bool(paper),
        "epochs": int(epochs),
        "seeds": seeds,
        "dataset": args.dataset,
        "K": int(args.K),
        "eps_marginal_frac_override": args.eps_marginal_frac,
        "calibrate_output": bool(args.calibrate_output),
        "calibration_include_2way": bool(args.calibration_include_2way),
        "calibration_passes": int(args.calibration_passes),
        "calibration_damping": float(args.calibration_damping),
        "suites": args.suite,
        "n_runs": int(len(rows)),
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "revision_ablations_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
