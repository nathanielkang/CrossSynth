"""
metrics.py - Evaluation metrics for CrossSynth experiments.

Metrics:
    1. ML Utility      - Train CatBoost/XGBoost on synthetic, test on real.
                          Report F1 (classification) or R2 (regression).
    2. Statistical Fidelity - Total variation distance on 1-way and 2-way
                          marginals between synthetic and real.
    3. Cross-Party Correlation - Mean absolute error of pairwise Pearson
                          correlations between synthetic union and real union.
    4. conditional_f1  - Unweighted mean of positive-class F1 within groups.
                          Not called by the experiment runner.

All functions work on numpy arrays.
"""

import warnings
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, r2_score


# ---------------------------------------------------------------------------
# 1. ML Utility
# ---------------------------------------------------------------------------

def ml_utility(X_synthetic, y_synthetic, X_test, y_test,
               task="classification", models=None):
    """
    Train ML models on synthetic data, evaluate on real test data.

    Parameters
    ----------
    X_synthetic : np.ndarray, shape (n_syn, d)
        Synthetic features for training.
    y_synthetic : np.ndarray, shape (n_syn,)
        Synthetic labels for training.
    X_test : np.ndarray, shape (n_test, d)
        Real test features.
    y_test : np.ndarray, shape (n_test,)
        Real test labels.
    task : str
        'classification' or 'regression'.
    models : list of str or None
        Which models to use. Default: ['catboost', 'xgboost'].

    Returns
    -------
    results : dict
        {model_name: {metric_name: value}}
    """
    if models is None:
        models = ["catboost", "xgboost"]

    results = {}

    for model_name in models:
        try:
            if model_name == "catboost":
                result = _eval_catboost(
                    X_synthetic, y_synthetic, X_test, y_test, task
                )
            elif model_name == "xgboost":
                result = _eval_xgboost(
                    X_synthetic, y_synthetic, X_test, y_test, task
                )
            else:
                warnings.warn(f"Unknown model: {model_name}")
                continue
            results[model_name] = result
        except Exception as e:
            warnings.warn(f"Error with {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results


def _eval_catboost(X_train, y_train, X_test, y_test, task):
    """Train and evaluate CatBoost."""
    if task == "classification":
        from catboost import CatBoostClassifier

        y_train_int = y_train.astype(int)
        y_test_int = y_test.astype(int)

        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_seed=42,
            thread_count=4,
        )
        model.fit(X_train, y_train_int)
        y_pred = model.predict(X_test).flatten().astype(int)

        return {
            "f1": float(f1_score(y_test_int, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test_int, y_pred)),
        }
    else:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_seed=42,
            thread_count=4,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            "r2": float(r2_score(y_test, y_pred)),
        }


def _eval_xgboost(X_train, y_train, X_test, y_test, task):
    """Train and evaluate XGBoost."""
    if task == "classification":
        from xgboost import XGBClassifier

        y_train_int = y_train.astype(int)
        y_test_int = y_test.astype(int)

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            verbosity=0,
            random_state=42,
            n_jobs=4,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train_int)
        y_pred = model.predict(X_test)

        return {
            "f1": float(f1_score(y_test_int, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test_int, y_pred)),
        }
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            verbosity=0,
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            "r2": float(r2_score(y_test, y_pred)),
        }


# ---------------------------------------------------------------------------
# 2. Statistical Fidelity - Marginal Distance
# ---------------------------------------------------------------------------

def marginal_distance_1way(X_synthetic, X_real, n_bins=20):
    """
    Total variation distance on 1-way marginals.

    For each column, compute normalized histograms of real and synthetic
    data, then compute TV distance = 0.5 * sum(|p - q|).
    Average across all columns.

    Parameters
    ----------
    X_synthetic : np.ndarray, shape (n_syn, d)
    X_real : np.ndarray, shape (n_real, d)
    n_bins : int

    Returns
    -------
    avg_tv : float
        Average total variation distance across columns.
    per_col_tv : list[float]
        TV distance per column.
    """
    d = X_real.shape[1]
    tvs = []

    for j in range(d):
        col_real = X_real[:, j]
        col_syn = X_synthetic[:, j]

        # Use common bin edges
        all_vals = np.concatenate([col_real, col_syn])
        vmin, vmax = all_vals.min(), all_vals.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        edges = np.linspace(vmin - 1e-6, vmax + 1e-6, n_bins + 1)

        hist_real, _ = np.histogram(col_real, bins=edges, density=True)
        hist_syn, _ = np.histogram(col_syn, bins=edges, density=True)

        # Normalize to probability distributions
        bin_width = edges[1] - edges[0]
        p = hist_real * bin_width
        q = hist_syn * bin_width

        # Ensure they sum to ~1
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

        tv = 0.5 * np.abs(p - q).sum()
        tvs.append(float(tv))

    return float(np.mean(tvs)), tvs


def marginal_distance_2way(X_synthetic, X_real, col_pairs=None, n_bins=10):
    """
    Total variation distance on 2-way marginals.

    Parameters
    ----------
    X_synthetic : np.ndarray, shape (n_syn, d)
    X_real : np.ndarray, shape (n_real, d)
    col_pairs : list of (int, int) or None
        If None, uses up to 50 random pairs.
    n_bins : int

    Returns
    -------
    avg_tv : float
    per_pair_tv : list[float]
    """
    from itertools import combinations

    d = X_real.shape[1]
    if col_pairs is None:
        all_pairs = list(combinations(range(d), 2))
        if len(all_pairs) > 50:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(all_pairs), 50, replace=False)
            col_pairs = [all_pairs[i] for i in idx]
        else:
            col_pairs = all_pairs

    tvs = []
    for (ci, cj) in col_pairs:
        all_i = np.concatenate([X_real[:, ci], X_synthetic[:, ci]])
        all_j = np.concatenate([X_real[:, cj], X_synthetic[:, cj]])

        def _edges(v):
            vmin, vmax = v.min(), v.max()
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            return np.linspace(vmin - 1e-6, vmax + 1e-6, n_bins + 1)

        ex = _edges(all_i)
        ey = _edges(all_j)

        h_real, _, _ = np.histogram2d(
            X_real[:, ci], X_real[:, cj], bins=[ex, ey]
        )
        h_syn, _, _ = np.histogram2d(
            X_synthetic[:, ci], X_synthetic[:, cj], bins=[ex, ey]
        )

        # Normalize
        p = h_real.flatten().astype(np.float64)
        q = h_syn.flatten().astype(np.float64)
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

        tv = 0.5 * np.abs(p - q).sum()
        tvs.append(float(tv))

    return float(np.mean(tvs)), tvs


# ---------------------------------------------------------------------------
# 3. Cross-Party Correlation Error
# ---------------------------------------------------------------------------

def correlation_error(X_synthetic, X_real):
    """
    Mean absolute error of pairwise Pearson correlations between
    synthetic and real data.

    Computes the full Pearson correlation matrix for both datasets,
    then returns the mean absolute difference of all unique pairs.

    Parameters
    ----------
    X_synthetic : np.ndarray, shape (n_syn, d)
    X_real : np.ndarray, shape (n_real, d)

    Returns
    -------
    mae_corr : float
        Mean absolute error of correlation coefficients.
    """
    d = X_real.shape[1]
    if d < 2:
        return 0.0

    # Compute correlation matrices
    # Handle constant columns gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_real = np.corrcoef(X_real, rowvar=False)
        corr_syn = np.corrcoef(X_synthetic, rowvar=False)

    # Replace NaN with 0 (constant columns produce NaN correlations)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_syn = np.nan_to_num(corr_syn, nan=0.0)

    # Extract upper triangle (unique pairs)
    mask = np.triu(np.ones((d, d), dtype=bool), k=1)
    real_vals = corr_real[mask]
    syn_vals = corr_syn[mask]

    mae = float(np.mean(np.abs(real_vals - syn_vals)))
    return mae


def conditional_f1(y_true, y_pred, group):
    """Unweighted mean of positive-class F1 inside each group.

    Positive class is 1. Each distinct value of ``group`` contributes one
    F1, and those scores are averaged with equal weight. The experiment
    runner does not call this function.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    group = np.asarray(group).reshape(-1)
    if not (y_true.shape[0] == y_pred.shape[0] == group.shape[0]):
        raise ValueError("y_true, y_pred, and group must have the same length.")
    if y_true.shape[0] == 0:
        return 0.0
    scores = []
    for g in np.unique(group):
        mask = group == g
        scores.append(float(
            f1_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0)
        ))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def conditional_ml_f1(X_synthetic, y_synthetic, X_test, y_test, group,
                      model_name="xgboost", random_state=42):
    """Train on synthetic data and average positive-class F1 across groups."""
    y_synthetic = np.asarray(y_synthetic).reshape(-1).astype(int)
    y_test = np.asarray(y_test).reshape(-1).astype(int)
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            verbosity=0,
            random_state=random_state,
            n_jobs=4,
            eval_metric="logloss",
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_seed=random_state,
            thread_count=4,
        )
    else:
        raise ValueError("model_name must be 'xgboost' or 'catboost'.")
    model.fit(X_synthetic, y_synthetic)
    prediction = np.asarray(model.predict(X_test)).reshape(-1).astype(int)
    return conditional_f1(y_test, prediction, group)


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics
# ---------------------------------------------------------------------------

def evaluate_all(X_synthetic, y_synthetic, X_real, y_real,
                 X_test, y_test, task="classification"):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    X_synthetic : np.ndarray
        Synthetic feature matrix (union of all party outputs).
    y_synthetic : np.ndarray
        Synthetic targets.
    X_real : np.ndarray
        Real training data (union of all parties).
    y_real : np.ndarray
        Real training targets.
    X_test : np.ndarray
        Real test features.
    y_test : np.ndarray
        Real test targets.
    task : str
        'classification' or 'regression'.

    Returns
    -------
    results : dict
        Nested dict with all metric values.
    """
    results = {}

    # ML Utility
    ml_res = ml_utility(X_synthetic, y_synthetic, X_test, y_test, task=task)
    results["ml_utility"] = ml_res

    # Statistical fidelity
    tv_1way, _ = marginal_distance_1way(X_synthetic, X_real)
    tv_2way, _ = marginal_distance_2way(X_synthetic, X_real)
    results["tv_1way"] = tv_1way
    results["tv_2way"] = tv_2way

    # Correlation error
    corr_err = correlation_error(X_synthetic, X_real)
    results["corr_error"] = corr_err

    return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # Fake real data
    n, d = 500, 8
    X_real = rng.randn(n, d).astype(np.float32)
    y_real = (X_real[:, 0] > 0).astype(np.float32)

    # Fake synthetic data (slightly shifted)
    X_syn = X_real + rng.normal(0, 0.3, X_real.shape).astype(np.float32)
    y_syn = (X_syn[:, 0] > 0).astype(np.float32)

    # Test set
    X_test = rng.randn(200, d).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.float32)

    # ML utility
    print("ML Utility:")
    ml_res = ml_utility(X_syn, y_syn, X_test, y_test, task="classification")
    for model_name, scores in ml_res.items():
        print(f"  {model_name}: {scores}")

    # Marginal distance
    tv1, _ = marginal_distance_1way(X_syn, X_real)
    tv2, _ = marginal_distance_2way(X_syn, X_real)
    print(f"\n1-way TV distance: {tv1:.4f}")
    print(f"2-way TV distance: {tv2:.4f}")

    # Correlation error
    ce = correlation_error(X_syn, X_real)
    print(f"Correlation error: {ce:.4f}")

    # Full evaluation
    print("\nFull evaluation:")
    results = evaluate_all(X_syn, y_syn, X_real, y_real, X_test, y_test)
    for k, v in results.items():
        print(f"  {k}: {v}")
