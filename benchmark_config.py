"""
benchmark_config.py - Unified benchmark settings for all 3 VLDB papers.

Settings aligned with:
  - TabDDPM (Kotelnikov et al., ICML 2023): evaluation protocol, seeds, classifiers
  - Kamino (Ge et al., PVLDB 2021): DP settings, metrics, 9 classifiers
  - PATE-GAN (Jordon et al., ICLR 2019): DP epsilon range, AUROC/AUPRC
  - CTGAN (Xu et al., NeurIPS 2019): benchmarking framework
"""

# ======================================================================
# Evaluation Protocol (matches TabDDPM Protocol 2 + Kamino)
# ======================================================================

# Number of seeds for synthetic data generation
N_GENERATION_SEEDS = 3  # Kamino uses 3; TabDDPM uses 5 (we use 3 for time)

# Number of seeds for downstream classifier training
N_CLASSIFIER_SEEDS = 5  # average over 5 classifier seeds per synthetic dataset

# Train/test split (matches Kamino)
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ======================================================================
# Downstream ML Models (matches Kamino's 9 classifiers)
# ======================================================================

# For classification tasks (Papers 1, 3)
CLASSIFICATION_MODELS = {
    "CatBoost": {"iterations": 500, "verbose": 0, "random_seed": 42},
    "XGBoost": {"n_estimators": 500, "verbosity": 0, "random_state": 42},
    "RandomForest": {"n_estimators": 100, "random_state": 42},
    "LogisticRegression": {"max_iter": 500, "random_state": 42},
    "MLP": {"hidden_layer_sizes": (256, 128), "max_iter": 200, "random_state": 42},
    "DecisionTree": {"max_depth": 28, "random_state": 42},
}

# For regression tasks (Paper 2)
REGRESSION_MODELS = {
    "CatBoost": {"iterations": 500, "verbose": 0, "random_seed": 42, "loss_function": "RMSE"},
    "XGBoost": {"n_estimators": 500, "verbosity": 0, "random_state": 42},
    "MLP": {"hidden_layer_sizes": (256, 128), "max_iter": 200, "random_state": 42},
}

# ======================================================================
# Diffusion Model Settings (matches TabDDPM search space)
# ======================================================================

DIFFUSION_TIMESTEPS = 1000     # standard DDPM
DIFFUSION_HIDDEN_DIM = 256     # MLP hidden dimension
DIFFUSION_N_LAYERS = 3         # number of hidden layers
DIFFUSION_DROPOUT = 0.0        # TabDDPM default
DIFFUSION_LR = 1e-3            # within TabDDPM range [1e-5, 3e-3]
DIFFUSION_BATCH_SIZE = 256     # TabDDPM: {256, 4096}
DIFFUSION_EPOCHS = 100         # ~10k-20k iterations for mid-size datasets

# ======================================================================
# Differential Privacy Settings (matches Kamino + PATE-GAN)
# ======================================================================

DP_DELTA = 1e-5                 # PATE-GAN standard
DP_EPSILONS = [0.1, 1.0, 10.0] # Kamino range: 0.1 to inf

# ======================================================================
# Paper 1 (CrossSynth) Specific
# ======================================================================

FEDSYNTH_K_VALUES = [3, 5]         # number of parties
FEDSYNTH_DATASETS = ["adult", "credit", "bank"]
FEDSYNTH_METHODS = ["fedsynth", "independent", "centralized", "privbayes"]
FEDSYNTH_N_MARGINAL_BINS = 20      # histogram bins for marginals

# ======================================================================
# Paper 2 (TabOversample) Specific
# ======================================================================

TABOVER_DATASETS = ["abalone", "california_housing", "bike_sharing",
                    "cpu_activity", "insurance", "house_16h"]
TABOVER_METHODS = ["None", "RandomOS", "SMOTER", "SMOGN",
                   "TabDDPM", "TabOversample"]

# ======================================================================
# Paper 3 (LLMSynth) Specific
# ======================================================================

LLMSYNTH_DATASETS = ["adult", "credit", "insurance"]
LLMSYNTH_METHODS = ["LLMSynth", "TabDDPM", "ManualConstraints", "PostHocRepair"]
LLMSYNTH_LLM_MODEL = "claude-sonnet-4-20250514"
LLMSYNTH_VALIDATION_THRESHOLD = 0.3  # discard LLM constraints with >30% violation
