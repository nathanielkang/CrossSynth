# CrossSynth

Reference implementation for **CrossSynth: Cross-Party Correlation-Preserving Federated Differentially Private Tabular Synthesis via Marginal-Conditioned Diffusion** (Neurocomputing submission). This repository contains **source code only**: training and evaluation scripts, marginal aggregation, and synthetic-data generators. It does **not** ship datasets (loaded at runtime via scikit-learn/OpenML) or precomputed experiment outputs.

**Repository:** https://github.com/nathanielkang/CrossSynth

## Requirements

- Python 3.10+
- See `requirements.txt` (PyTorch, NumPy, pandas, scikit-learn, CatBoost, SciPy, tqdm, tabulate).

Install:

```bash
pip install -r requirements.txt
```

## Quick test

Smoke run on a small grid (few datasets/methods, short training):

```bash
python run_experiments.py --quick
```

## Full experiments

Default grid (multiple datasets, partition modes, \(K\), \(\varepsilon\), methods, seeds) writes CSV and LaTeX tables under `results/` (created locally; not tracked in git):

```bash
python run_experiments.py
```

Filter examples:

```bash
python run_experiments.py --dataset adult --method fedsynth
python run_experiments.py --partition correlated --seeds 3 --epochs 100
```

## Package layout

| File | Role |
|------|------|
| `run_experiments.py` | CLI entry point; grid search and metrics aggregation |
| `synthesizer.py` | CrossSynth, independent/c centralized TabDDPM-style diffusion, PrivBayes-style baseline |
| `marginals.py` | Noisy marginal counts, aggregation, encoding for conditioning |
| `datasets.py` | OpenML loaders, \(K\)-way partitioning (random / correlated), preprocessing |
| `metrics.py` | ML utility (CatBoost F1), marginal TVD, association error |
| `benchmark_config.py` | Shared hyperparameters and grid definitions |

Datasets are **downloaded automatically** through `sklearn.datasets.fetch_openml` where applicable; no proprietary files are included.

## Citation

If you use this code, please cite the paper (Neurocomputing, when available) and this repository.

## License

Code is provided for research purposes. See the repository license file if present.
