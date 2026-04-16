"""
datasets.py - Load and preprocess tabular datasets for CrossSynth experiments.

Datasets (all via sklearn/OpenML):
    1. Adult (Census income)  - binary classification, ~32K rows, 14 features
    2. Credit-G (German)      - binary classification, 1000 rows, 20 features
    3. Bank Marketing          - binary classification, ~45K rows, 16 features

Functions:
    - load_*()           : Load individual datasets as DataFrames
    - get_dataset(name)  : Registry-based loader
    - partition_data()   : Split into K disjoint parties (random or correlated)
    - preprocess()       : Label-encode categoricals, standardize numericals

Designed to run on CPU (32 GB RAM). All datasets auto-download.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_encode_categoricals(df, cat_cols):
    """Label-encode categorical columns in-place and return encoders dict."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _detect_cat_cols(df):
    """Detect categorical columns (object or category dtype)."""
    return df.select_dtypes(include=["category", "object"]).columns.tolist()


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_adult():
    """
    Adult Census Income dataset - predict whether income >50K.
    ~32K rows, 14 features (mix of categorical and numerical).
    """
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.copy()

    # Standardize target to binary 0/1
    target = target.astype(str).str.strip().str.replace(".", "", regex=False)
    target = (target.isin([">50K", "1", ">50K."])).astype(int).values

    # Drop rows with NaN
    mask = df.notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    target = target[mask]

    cat_cols = _detect_cat_cols(df)
    return df, target, cat_cols, "Adult"


def load_credit():
    """
    German Credit dataset - predict credit risk (good/bad).
    1000 rows, 20 features.
    """
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.copy()

    # Target: 'good' -> 1, 'bad' -> 0
    target = (target.astype(str).str.strip() == "good").astype(int).values

    mask = df.notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    target = target[mask]

    cat_cols = _detect_cat_cols(df)
    return df, target, cat_cols, "Credit"


def load_bank():
    """
    Bank Marketing dataset - predict term deposit subscription.
    ~45K rows, 16 features.
    """
    data = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.copy()

    # Target: '2' or 'yes' -> 1, '1' or 'no' -> 0
    target_str = target.astype(str).str.strip().str.lower()
    target = target_str.isin(["2", "yes"]).astype(int).values

    mask = df.notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    target = target[mask]

    cat_cols = _detect_cat_cols(df)
    return df, target, cat_cols, "Bank"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df, target, cat_cols, random_state=42):
    """
    Preprocess a dataset: label-encode categoricals, standardize numericals,
    and split into train/test (70/30).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    target : np.ndarray
        Binary target array.
    cat_cols : list[str]
        Categorical column names.
    random_state : int
        Random seed for split.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test  (np.ndarray, float32)
        columns      : list of column names
        cat_indices  : list of int indices for categorical columns
        num_indices  : list of int indices for numerical columns
        scaler       : fitted StandardScaler (for numerical cols)
        encoders     : dict of LabelEncoders (for categorical cols)
    """
    df = df.copy()
    df, encoders = _label_encode_categoricals(df, cat_cols)

    columns = list(df.columns)
    cat_indices = [columns.index(c) for c in cat_cols]
    num_indices = [i for i in range(len(columns)) if i not in cat_indices]

    X = df.values.astype(np.float32)
    y = target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # Standardize numerical features (fit on train only)
    scaler = StandardScaler()
    if num_indices:
        X_train[:, num_indices] = scaler.fit_transform(
            X_train[:, num_indices]
        )
        X_test[:, num_indices] = scaler.transform(
            X_test[:, num_indices]
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "columns": columns,
        "cat_indices": cat_indices,
        "num_indices": num_indices,
        "scaler": scaler,
        "encoders": encoders,
    }


# ---------------------------------------------------------------------------
# Partition column mapping (for correlated partitioning)
# ---------------------------------------------------------------------------

_PARTITION_COLUMNS = {
    "adult": "age",
    "credit": "credit_amount",
    "bank": "age",
}


# ---------------------------------------------------------------------------
# Partitioning into K parties
# ---------------------------------------------------------------------------

def partition_data(X, y, K, mode="random", random_state=42,
                   partition_col_idx=0):
    """
    Split data into K disjoint parties.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    K : int
        Number of parties.
    mode : str
        'random'     - uniformly random partition
        'correlated' - party k gets samples from the k-th quantile range
                       of the column at partition_col_idx, creating
                       heterogeneous parties with different distributions.
    random_state : int
        Seed for reproducibility.
    partition_col_idx : int
        Column index to use for correlated partitioning (ignored for random).

    Returns
    -------
    parties : list of dict
        Each dict has 'X' (np.ndarray) and 'y' (np.ndarray).
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]

    if mode == "random":
        indices = rng.permutation(n)
        splits = np.array_split(indices, K)

    elif mode == "correlated":
        # Quantile-based split on the specified column.
        # Party k gets rows where the column value falls in the k-th
        # quantile range, creating parties with DIFFERENT distributions.
        col_values = X[:, partition_col_idx]
        quantile_edges = np.quantile(col_values, np.linspace(0, 1, K + 1))

        splits = []
        assigned = np.zeros(n, dtype=bool)

        for k in range(K):
            low = quantile_edges[k]
            high = quantile_edges[k + 1]
            if k == K - 1:
                # Last partition: include upper boundary
                mask = (col_values >= low) & (col_values <= high) & (~assigned)
            else:
                mask = (col_values >= low) & (col_values < high) & (~assigned)

            idx = np.where(mask)[0]

            # If empty (e.g. many duplicates at boundary), fall back to
            # unassigned rows split evenly
            if len(idx) == 0:
                unassigned = np.where(~assigned)[0]
                if len(unassigned) > 0:
                    chunk_size = max(1, len(unassigned) // (K - k))
                    idx = unassigned[:chunk_size]

            assigned[idx] = True
            splits.append(idx)

        # Assign any remaining unassigned rows to the last party
        remaining = np.where(~assigned)[0]
        if len(remaining) > 0:
            splits[-1] = np.concatenate([splits[-1], remaining])

    else:
        raise ValueError(f"Unknown partition mode: '{mode}'")

    parties = []
    for idx in splits:
        parties.append({
            "X": X[idx].copy(),
            "y": y[idx].copy(),
        })

    return parties


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "adult": load_adult,
    "credit": load_credit,
    "bank": load_bank,
}


def get_dataset(name, random_state=42):
    """
    Load and preprocess a dataset by name.

    Parameters
    ----------
    name : str
        One of: 'adult', 'credit', 'bank'
    random_state : int
        Seed for train/test split.

    Returns
    -------
    dict  (X_train, y_train, X_test, y_test, columns, cat_indices, ...,
           partition_col_idx, partition_col_name)
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(_DATASET_LOADERS.keys())}"
        )
    print(f"[datasets] Loading {key} ...")
    df, target, cat_cols, display_name = _DATASET_LOADERS[key]()
    result = preprocess(df, target, cat_cols, random_state=random_state)
    result["name"] = display_name

    # Resolve partition column for correlated partitioning
    partition_col_name = _PARTITION_COLUMNS.get(key, None)
    if partition_col_name and partition_col_name in result["columns"]:
        result["partition_col_idx"] = result["columns"].index(partition_col_name)
        result["partition_col_name"] = partition_col_name
    else:
        # Fallback: first numerical column
        result["partition_col_idx"] = (
            result["num_indices"][0] if result["num_indices"] else 0
        )
        result["partition_col_name"] = result["columns"][result["partition_col_idx"]]

    print(f"  -> {display_name}: train={result['X_train'].shape}, "
          f"test={result['X_test'].shape}, "
          f"cat={len(result['cat_indices'])}, "
          f"num={len(result['num_indices'])}, "
          f"partition_col={result['partition_col_name']}"
          f"(idx={result['partition_col_idx']})")
    return result


def get_all_datasets(random_state=42):
    """Load all benchmark datasets. Returns list of dicts."""
    datasets = []
    for key in _DATASET_LOADERS:
        try:
            ds = get_dataset(key, random_state=random_state)
            datasets.append(ds)
        except Exception as e:
            print(f"  [WARN] Failed to load {key}: {e}")
    return datasets


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = get_all_datasets()
    print(f"\nLoaded {len(datasets)} datasets successfully.")
    for ds in datasets:
        print(f"  {ds['name']:20s}  train={ds['X_train'].shape}  "
              f"test={ds['X_test'].shape}  "
              f"cat_idx={ds['cat_indices']}  "
              f"partition_col={ds['partition_col_name']}"
              f"(idx={ds['partition_col_idx']})")

    # Test random partitioning
    ds = datasets[0]
    parties = partition_data(ds["X_train"], ds["y_train"], K=3, mode="random")
    print(f"\nRandom partition of {ds['name']} into {len(parties)} parties:")
    for i, p in enumerate(parties):
        print(f"  Party {i}: X={p['X'].shape}, "
              f"class_dist={np.bincount(p['y'].astype(int))}")

    # Test correlated partitioning
    pcol = ds["partition_col_idx"]
    parties_corr = partition_data(
        ds["X_train"], ds["y_train"], K=3, mode="correlated",
        partition_col_idx=pcol
    )
    print(f"\nCorrelated partition of {ds['name']} "
          f"(col={ds['partition_col_name']}, idx={pcol}):")
    for i, p in enumerate(parties_corr):
        col_vals = p['X'][:, pcol]
        print(f"  Party {i}: X={p['X'].shape}, "
              f"col_range=[{col_vals.min():.2f}, {col_vals.max():.2f}], "
              f"class_dist={np.bincount(p['y'].astype(int))}")
