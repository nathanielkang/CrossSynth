"""
marginals.py - Marginal computation, DP noise injection, and secure aggregation.

Implements:
    - 1-way marginals: histogram per column (numericals discretized into bins)
    - 2-way marginals: joint histograms for column pairs
    - Gaussian mechanism for differential privacy
    - Secure aggregation simulation: average noisy marginals across parties
    - Marginal encoding: flatten aggregated marginals into a fixed-size vector

Used by CrossSynth to condition the diffusion model on cross-party statistics.
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# 1-way marginals
# ---------------------------------------------------------------------------

def compute_1way_marginals(X, n_bins=20):
    """
    Compute 1-way marginal histograms for each column.

    Each column's values are discretized into `n_bins` equal-width bins
    spanning [col_min - eps, col_max + eps]. The histogram is normalized
    to sum to 1 (probability distribution).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix.
    n_bins : int
        Number of bins per column.

    Returns
    -------
    marginals : list of dict
        Each dict has:
            'hist'  : np.ndarray, shape (n_bins,) - normalized histogram
            'edges' : np.ndarray, shape (n_bins+1,) - bin edges
            'col'   : int - column index
    """
    n, d = X.shape
    marginals = []

    for col_idx in range(d):
        col_data = X[:, col_idx]
        col_min, col_max = col_data.min(), col_data.max()

        # Add small epsilon to avoid zero-width bins
        if col_min == col_max:
            col_min -= 0.5
            col_max += 0.5

        edges = np.linspace(col_min - 1e-6, col_max + 1e-6, n_bins + 1)
        hist, _ = np.histogram(col_data, bins=edges)
        hist = hist.astype(np.float64)

        # Normalize to probability distribution
        total = hist.sum()
        if total > 0:
            hist = hist / total

        marginals.append({
            "hist": hist,
            "edges": edges,
            "col": col_idx,
        })

    return marginals


# ---------------------------------------------------------------------------
# 2-way marginals
# ---------------------------------------------------------------------------

def compute_2way_marginals(X, col_pairs=None, n_bins=10):
    """
    Compute 2-way joint marginal histograms for selected column pairs.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix.
    col_pairs : list of (int, int) or None
        Column pairs to compute. If None, uses all pairs (up to 50).
    n_bins : int
        Number of bins per dimension in the joint histogram.

    Returns
    -------
    marginals_2way : list of dict
        Each dict has:
            'hist'  : np.ndarray, shape (n_bins, n_bins) - normalized joint hist
            'edges_x' : np.ndarray
            'edges_y' : np.ndarray
            'cols' : (int, int)
    """
    n, d = X.shape

    if col_pairs is None:
        # Use all pairs, but limit to avoid combinatorial explosion
        all_pairs = list(combinations(range(d), 2))
        if len(all_pairs) > 50:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(all_pairs), 50, replace=False)
            col_pairs = [all_pairs[i] for i in idx]
        else:
            col_pairs = all_pairs

    marginals_2way = []

    for (ci, cj) in col_pairs:
        xi, xj = X[:, ci], X[:, cj]

        # Bin edges
        def _edges(v):
            vmin, vmax = v.min(), v.max()
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            return np.linspace(vmin - 1e-6, vmax + 1e-6, n_bins + 1)

        ex = _edges(xi)
        ey = _edges(xj)

        hist, _, _ = np.histogram2d(xi, xj, bins=[ex, ey])
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist = hist / total

        marginals_2way.append({
            "hist": hist,
            "edges_x": ex,
            "edges_y": ey,
            "cols": (ci, cj),
        })

    return marginals_2way


# ---------------------------------------------------------------------------
# Gaussian mechanism for DP
# ---------------------------------------------------------------------------

def _calibrate_gaussian_noise(sensitivity, epsilon, delta=1e-5):
    """
    Calibrate Gaussian noise standard deviation for (epsilon, delta)-DP.

    Uses the analytic Gaussian mechanism:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Parameters
    ----------
    sensitivity : float
        L2 sensitivity of the query.
    epsilon : float
        Privacy budget.
    delta : float
        Probability of privacy breach.

    Returns
    -------
    sigma : float
        Standard deviation of Gaussian noise.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    return sigma


def add_dp_noise_1way(marginals, n_samples, epsilon, delta=1e-5):
    """
    Add calibrated Gaussian noise to 1-way marginal histograms for DP.

    The sensitivity of a normalized histogram with n_samples entries is
    1/n_samples per bin (adding or removing one record changes one bin
    by at most 1/n).

    Parameters
    ----------
    marginals : list of dict
        Output of compute_1way_marginals().
    n_samples : int
        Number of samples used to compute the marginals (for sensitivity).
    epsilon : float
        Privacy budget for these marginals.
    delta : float
        DP delta parameter.

    Returns
    -------
    noisy_marginals : list of dict
        Same structure with noise added to 'hist'.
    """
    sensitivity = 1.0 / max(n_samples, 1)
    sigma = _calibrate_gaussian_noise(sensitivity, epsilon, delta)

    noisy = []
    for m in marginals:
        hist_noisy = m["hist"].copy() + np.random.normal(0, sigma, size=m["hist"].shape)
        # Clip negatives and re-normalize
        hist_noisy = np.maximum(hist_noisy, 0.0)
        total = hist_noisy.sum()
        if total > 0:
            hist_noisy = hist_noisy / total
        else:
            # Uniform fallback if all noise killed the signal
            hist_noisy = np.ones_like(hist_noisy) / len(hist_noisy)

        noisy.append({
            "hist": hist_noisy,
            "edges": m["edges"].copy(),
            "col": m["col"],
        })
    return noisy


def add_dp_noise_2way(marginals_2way, n_samples, epsilon, delta=1e-5):
    """
    Add calibrated Gaussian noise to 2-way marginal histograms for DP.

    Parameters
    ----------
    marginals_2way : list of dict
        Output of compute_2way_marginals().
    n_samples : int
    epsilon : float
    delta : float

    Returns
    -------
    noisy_marginals : list of dict
    """
    sensitivity = 1.0 / max(n_samples, 1)
    sigma = _calibrate_gaussian_noise(sensitivity, epsilon, delta)

    noisy = []
    for m in marginals_2way:
        hist_noisy = m["hist"].copy() + np.random.normal(0, sigma, size=m["hist"].shape)
        hist_noisy = np.maximum(hist_noisy, 0.0)
        total = hist_noisy.sum()
        if total > 0:
            hist_noisy = hist_noisy / total
        else:
            hist_noisy = np.ones_like(hist_noisy) / hist_noisy.size

        noisy.append({
            "hist": hist_noisy,
            "edges_x": m["edges_x"].copy(),
            "edges_y": m["edges_y"].copy(),
            "cols": m["cols"],
        })
    return noisy


# ---------------------------------------------------------------------------
# Secure aggregation (simulated)
# ---------------------------------------------------------------------------

def aggregate_marginals_1way(all_party_marginals):
    """
    Securely aggregate 1-way marginals from multiple parties.

    In a real system, this would use secure aggregation protocols.
    Here we simulate it by averaging the noisy marginals.

    Parameters
    ----------
    all_party_marginals : list of list of dict
        all_party_marginals[k] is the list of 1-way marginals from party k.

    Returns
    -------
    aggregated : list of dict
        Averaged 1-way marginals (one per column).
    """
    K = len(all_party_marginals)
    n_cols = len(all_party_marginals[0])

    aggregated = []
    for col_idx in range(n_cols):
        # Average histograms across parties
        hists = [all_party_marginals[k][col_idx]["hist"] for k in range(K)]
        avg_hist = np.mean(hists, axis=0)

        # Re-normalize
        total = avg_hist.sum()
        if total > 0:
            avg_hist = avg_hist / total

        # Use edges from the first party (they should be similar)
        # In practice, parties would agree on bin edges beforehand
        aggregated.append({
            "hist": avg_hist,
            "edges": all_party_marginals[0][col_idx]["edges"].copy(),
            "col": col_idx,
        })

    return aggregated


def aggregate_marginals_2way(all_party_marginals_2way):
    """
    Securely aggregate 2-way marginals from multiple parties.

    Parameters
    ----------
    all_party_marginals_2way : list of list of dict

    Returns
    -------
    aggregated : list of dict
    """
    K = len(all_party_marginals_2way)
    n_pairs = len(all_party_marginals_2way[0])

    aggregated = []
    for pair_idx in range(n_pairs):
        hists = [all_party_marginals_2way[k][pair_idx]["hist"] for k in range(K)]
        avg_hist = np.mean(hists, axis=0)
        total = avg_hist.sum()
        if total > 0:
            avg_hist = avg_hist / total

        aggregated.append({
            "hist": avg_hist,
            "edges_x": all_party_marginals_2way[0][pair_idx]["edges_x"].copy(),
            "edges_y": all_party_marginals_2way[0][pair_idx]["edges_y"].copy(),
            "cols": all_party_marginals_2way[0][pair_idx]["cols"],
        })

    return aggregated


# ---------------------------------------------------------------------------
# Marginal encoding (for conditioning the diffusion model)
# ---------------------------------------------------------------------------

def encode_marginals(marginals_1way, marginals_2way=None, max_dim=256):
    """
    Encode marginal histograms into a fixed-size conditioning vector.

    Concatenates all 1-way histograms (and optionally flattened 2-way hists),
    then truncates or pads to `max_dim`.

    Parameters
    ----------
    marginals_1way : list of dict
        1-way marginals with 'hist' field.
    marginals_2way : list of dict or None
        Optional 2-way marginals.
    max_dim : int
        Target dimensionality of the output vector.

    Returns
    -------
    vec : np.ndarray, shape (max_dim,)
        Normalized conditioning vector.
    """
    parts = []

    # Concatenate 1-way histograms
    for m in marginals_1way:
        parts.append(m["hist"].flatten())

    # Optionally add 2-way histograms
    if marginals_2way is not None:
        for m in marginals_2way:
            parts.append(m["hist"].flatten())

    vec = np.concatenate(parts).astype(np.float32)

    # Truncate or pad to max_dim
    if len(vec) > max_dim:
        vec = vec[:max_dim]
    elif len(vec) < max_dim:
        vec = np.concatenate([vec, np.zeros(max_dim - len(vec), dtype=np.float32)])

    # Standardize to zero mean, unit variance.
    # This ensures the conditioning vector has values on the same scale as
    # the standardized input features (~N(0,1)), making the conditioning
    # signal strong enough for the denoising network to use effectively.
    # (Previous L2-normalization gave element magnitudes ~1/sqrt(dim) ~0.06,
    # which was too small relative to input features and time embeddings.)
    mean_val = vec.mean()
    std_val = vec.std()
    if std_val > 1e-8:
        vec = (vec - mean_val) / std_val
    else:
        vec = vec - mean_val

    return vec


def compute_shared_bin_edges(parties, n_bins=20):
    """
    Compute shared bin edges across all parties for consistent marginals.

    Parameters
    ----------
    parties : list of dict
        Each party dict has 'X' (np.ndarray).
    n_bins : int
        Number of bins per column.

    Returns
    -------
    all_edges : list of np.ndarray
        all_edges[col] has shape (n_bins+1,).
    """
    # Pool min/max across parties
    d = parties[0]["X"].shape[1]
    all_edges = []

    for col_idx in range(d):
        col_min = min(p["X"][:, col_idx].min() for p in parties)
        col_max = max(p["X"][:, col_idx].max() for p in parties)
        if col_min == col_max:
            col_min -= 0.5
            col_max += 0.5
        edges = np.linspace(col_min - 1e-6, col_max + 1e-6, n_bins + 1)
        all_edges.append(edges)

    return all_edges


def compute_1way_marginals_shared(X, shared_edges):
    """
    Compute 1-way marginals using pre-agreed shared bin edges.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    shared_edges : list of np.ndarray
        One set of edges per column.

    Returns
    -------
    marginals : list of dict
    """
    d = X.shape[1]
    marginals = []

    for col_idx in range(d):
        edges = shared_edges[col_idx]
        hist, _ = np.histogram(X[:, col_idx], bins=edges)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist = hist / total

        marginals.append({
            "hist": hist,
            "edges": edges,
            "col": col_idx,
        })

    return marginals


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 5).astype(np.float32)

    # 1-way
    m1 = compute_1way_marginals(X, n_bins=20)
    print(f"1-way marginals: {len(m1)} columns, bins per col = {m1[0]['hist'].shape}")

    # Add DP noise
    m1_noisy = add_dp_noise_1way(m1, n_samples=1000, epsilon=1.0)
    print(f"Noisy 1-way TV distance (col 0): "
          f"{0.5 * np.abs(m1[0]['hist'] - m1_noisy[0]['hist']).sum():.4f}")

    # 2-way
    m2 = compute_2way_marginals(X, col_pairs=[(0, 1), (2, 3)], n_bins=10)
    print(f"2-way marginals: {len(m2)} pairs, shape = {m2[0]['hist'].shape}")

    # Aggregation
    m1_party0 = add_dp_noise_1way(m1, 500, epsilon=1.0)
    m1_party1 = add_dp_noise_1way(m1, 500, epsilon=1.0)
    agg = aggregate_marginals_1way([m1_party0, m1_party1])
    print(f"Aggregated marginals: {len(agg)} columns")

    # Encoding
    vec = encode_marginals(agg, max_dim=128)
    print(f"Marginal encoding vector: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
