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


def column_domain_sizes(shared_edges):
    """Public per-column domain sizes from the agreed histogram edges.

    ``|dom_i|`` is the number of shared bins. Equal-width columns therefore
    tie, and pair ranking falls through to column order.
    """
    return [max(int(len(edges) - 1), 1) for edges in shared_edges]


def select_2way_pairs(n_cols, M=50, selection="domain_size",
                      domain_sizes=None, rng_seed=42):
    """Choose up to ``M`` two-way pairs.

    ``domain_size`` ranks pairs by ``|dom_i| * |dom_j|`` ascending and keeps
    the first ``M``. ``random`` draws ``M`` pairs from a local ``RandomState``
    so the global NumPy stream used by the one-way noise is left alone.
    One-way marginals are not returned here; the caller always keeps them.
    """
    if n_cols < 2 or M <= 0:
        return []
    all_pairs = list(combinations(range(n_cols), 2))
    if selection == "domain_size":
        if domain_sizes is None:
            domain_sizes = [1] * n_cols
        if len(domain_sizes) != n_cols:
            raise ValueError("domain_sizes must have one entry per column.")
        ranked = sorted(
            all_pairs,
            key=lambda ij: (
                float(domain_sizes[ij[0]]) * float(domain_sizes[ij[1]]),
                ij[0],
                ij[1],
            ),
        )
        return ranked[:M]
    if selection == "random":
        if len(all_pairs) <= M:
            return list(all_pairs)
        rng = np.random.RandomState(rng_seed)
        chosen_idx = rng.choice(len(all_pairs), size=M, replace=False)
        chosen = [all_pairs[int(i)] for i in chosen_idx]
        chosen.sort()
        return chosen
    raise ValueError(
        f"Unknown selection '{selection}'. Choose from: domain_size, random."
    )


def compute_2way_marginals_shared(X, col_pairs, shared_edges):
    """Joint histograms on the same shared edges as the one-way marginals."""
    marginals_2way = []
    if not col_pairs:
        return marginals_2way
    for (ci, cj) in col_pairs:
        ex = shared_edges[ci]
        ey = shared_edges[cj]
        hist, _, _ = np.histogram2d(X[:, ci], X[:, cj], bins=[ex, ey])
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist = hist / total
        marginals_2way.append({
            "hist": hist,
            "edges_x": np.asarray(ex).copy(),
            "edges_y": np.asarray(ey).copy(),
            "cols": (int(ci), int(cj)),
        })
    return marginals_2way


def compute_1way_counts_shared(X, shared_edges):
    """Compute unnormalised one-way counts on public shared bin edges."""
    counts = []
    for col_idx, edges in enumerate(shared_edges):
        hist, _ = np.histogram(X[:, col_idx], bins=edges)
        counts.append({
            "hist": hist.astype(np.float64),
            "edges": np.asarray(edges).copy(),
            "col": int(col_idx),
        })
    return counts


def compute_2way_counts_shared(X, col_pairs, shared_edges):
    """Compute unnormalised two-way counts on public shared bin edges."""
    counts = []
    for ci, cj in col_pairs:
        ex = shared_edges[ci]
        ey = shared_edges[cj]
        hist, _, _ = np.histogram2d(X[:, ci], X[:, cj], bins=[ex, ey])
        counts.append({
            "hist": hist.astype(np.float64),
            "edges_x": np.asarray(ex).copy(),
            "edges_y": np.asarray(ey).copy(),
            "cols": (int(ci), int(cj)),
        })
    return counts


def maybe_round_count_hists(marginals, round_counts=False):
    """Round noisy histogram vectors to integers before aggregation.

    The Gaussian draw is continuous. Rounding is deterministic
    post-processing so a later modular sum can cancel the masks.
    ``round_counts=False`` returns the float histograms unchanged.
    ``CrossSynthGenerator.fit`` turns rounding on by default.
    """
    if not round_counts:
        return marginals
    rounded = []
    for m in marginals:
        item = dict(m)
        item["hist"] = np.rint(np.asarray(m["hist"], dtype=np.float64))
        rounded.append(item)
    return rounded


def add_dp_noise_count_queries(marginals_1way, marginals_2way, epsilon,
                               delta=1e-5, replace_one=True, rng=None):
    """Release one concatenated noisy count workload.

    Every record contributes to one bin in each query. Under replace-one
    adjacency, changing a record changes at most two bins per query, so the
    L2 sensitivity of the concatenated workload is ``sqrt(2 * |Q|)``. Under
    add/remove adjacency it is ``sqrt(|Q|)``. One Gaussian draw is calibrated
    to the complete workload; the privacy budget is not spent once per query.

    The returned histograms remain in count space. Clipping, normalisation,
    integer rounding, and secure summation are downstream post-processing.
    """
    marginals_2way = list(marginals_2way or [])
    query_count = len(marginals_1way) + len(marginals_2way)
    if query_count <= 0:
        raise ValueError("At least one marginal query is required.")
    sensitivity = np.sqrt((2.0 if replace_one else 1.0) * query_count)
    sigma = _calibrate_gaussian_noise(sensitivity, epsilon, delta)
    if rng is None:
        rng = np.random

    def _noisy_copy(items):
        released = []
        for item in items:
            copied = dict(item)
            hist = np.asarray(item["hist"], dtype=np.float64)
            copied["hist"] = hist + rng.normal(0.0, sigma, size=hist.shape)
            released.append(copied)
        return released

    return (
        _noisy_copy(marginals_1way),
        _noisy_copy(marginals_2way),
        float(sigma),
        float(sensitivity),
    )


def scale_like_encoded_marginal(raw, encoded):
    """Map a control vector with the encoded marginal's mean and std.

    ``encode_marginals`` standardizes the histogram vector. Applying that
    vector's own mean and std puts a same-length control on the encoder
    output scale. A constant is not standardized with its own variance:
    that variance is zero and the map would be the zero vector.
    """
    raw = np.asarray(raw, dtype=np.float32)
    ref = np.asarray(encoded, dtype=np.float32)
    mean_val = ref.mean()
    std_val = ref.std()
    if std_val > 1e-8:
        return ((raw - mean_val) / std_val).astype(np.float32)
    return (raw - mean_val).astype(np.float32)


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
    # Legacy normalized-histogram path. Sensitivity stays 1/n_samples.
    # The manuscript count workload uses add_dp_noise_count_queries.
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
    # Legacy normalized-histogram path, same 1/n_samples scale as
    # add_dp_noise_1way. The count workload does not use this function.
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


def secure_sum_integer_vectors(vectors, modulus=2_147_483_647,
                               rng_seed=42):
    """Simulate serverless additive secret sharing over ``Z_modulus``.

    Each sender makes one share for every receiver. Receivers broadcast their
    partial sums, and the partial sums reconstruct only the aggregate. The
    function returns the centered signed representative of that aggregate.
    It is a protocol simulator, not a cryptographic transport implementation.
    """
    if len(vectors) < 2:
        raise ValueError("Secure summation requires at least two parties.")
    if modulus <= 2:
        raise ValueError("modulus must be greater than 2.")
    arrays = [np.asarray(v, dtype=np.int64).reshape(-1) for v in vectors]
    length = arrays[0].size
    if any(v.size != length for v in arrays):
        raise ValueError("All secure-sum vectors must have the same length.")

    plain = np.sum(np.stack(arrays, axis=0), axis=0, dtype=np.int64)
    if np.any(np.abs(plain) >= modulus // 2):
        raise ValueError("modulus is too small for centered reconstruction.")

    rng = np.random.RandomState(rng_seed)
    receiver_partials = [np.zeros(length, dtype=np.int64)
                         for _ in arrays]
    for sender_idx, vector in enumerate(arrays):
        random_shares = []
        for _ in range(len(arrays) - 1):
            random_shares.append(
                rng.randint(0, modulus, size=length, dtype=np.int64)
            )
        residual = np.mod(
            np.mod(vector, modulus)
            - sum(random_shares, np.zeros(length, dtype=np.int64)),
            modulus,
        )
        shares = random_shares + [residual]
        # Rotate the residual receiver so one participant does not always get
        # every residual share. This does not change reconstruction.
        shares = shares[-sender_idx:] + shares[:-sender_idx] if sender_idx else shares
        for receiver_idx, share in enumerate(shares):
            receiver_partials[receiver_idx] = np.mod(
                receiver_partials[receiver_idx] + share, modulus
            )

    reconstructed = np.zeros(length, dtype=np.int64)
    for partial in receiver_partials:
        reconstructed = np.mod(reconstructed + partial, modulus)
    signed = reconstructed.copy()
    signed[signed > modulus // 2] -= modulus
    if not np.array_equal(signed, plain):
        raise AssertionError("Additive-share reconstruction failed.")
    return signed


def aggregate_count_queries(all_party_1way, all_party_2way=None,
                            secure=False, modulus=2_147_483_647,
                            rng_seed=42):
    """Sum noisy count queries and convert each query to a probability table.

    Negative noisy cells are clipped only after cross-party aggregation. When
    ``secure`` is true, inputs must already have integer-valued histograms and
    the sum is reconstructed with :func:`secure_sum_integer_vectors`.
    """
    if not all_party_1way:
        raise ValueError("At least one party is required.")
    all_party_2way = all_party_2way or [[] for _ in all_party_1way]

    def _aggregate_query_group(groups, key_kind):
        if not groups or not groups[0]:
            return []
        n_queries = len(groups[0])
        if any(len(group) != n_queries for group in groups):
            raise ValueError("Every party must release the same query set.")
        result = []
        for query_idx in range(n_queries):
            vectors = [group[query_idx]["hist"].reshape(-1) for group in groups]
            if secure:
                if any(not np.allclose(v, np.rint(v)) for v in vectors):
                    raise ValueError(
                        "Secure modular summation requires rounded integer counts."
                    )
                summed = secure_sum_integer_vectors(
                    [np.rint(v).astype(np.int64) for v in vectors],
                    modulus=modulus,
                    rng_seed=rng_seed + query_idx,
                ).astype(np.float64)
            else:
                summed = np.sum(np.stack(vectors, axis=0), axis=0)
            shape = groups[0][query_idx]["hist"].shape
            summed = np.maximum(summed.reshape(shape), 0.0)
            total = float(summed.sum())
            if total > 0:
                probability = summed / total
            else:
                probability = np.full(shape, 1.0 / summed.size)
            copied = dict(groups[0][query_idx])
            copied["hist"] = probability
            copied["query_kind"] = key_kind
            result.append(copied)
        return result

    return (
        _aggregate_query_group(all_party_1way, "one_way"),
        _aggregate_query_group(all_party_2way, "two_way"),
    )


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


def compute_shared_bin_edges(parties, n_bins=20, categorical_indices=None):
    """
    Compute shared bin edges across all parties for consistent marginals.

    Parameters
    ----------
    parties : list of dict
        Each party dict has 'X' (np.ndarray).
    n_bins : int
        Number of bins per column.
    categorical_indices : iterable[int] or None
        Integer-coded categorical columns. These receive one unit-width bin
        per observed category rather than ``n_bins`` numerical bins.

    Returns
    -------
    all_edges : list of np.ndarray
        all_edges[col] has shape (n_bins+1,).
    """
    # Pool min/max across parties. The runner supplies integer-coded columns
    # from the dataset schema, including the jointly synthesized binary label.
    d = parties[0]["X"].shape[1]
    all_edges = []
    categorical_indices = set(categorical_indices or [])

    for col_idx in range(d):
        col_min = min(p["X"][:, col_idx].min() for p in parties)
        col_max = max(p["X"][:, col_idx].max() for p in parties)
        if col_idx in categorical_indices:
            low = int(np.floor(col_min))
            high = int(np.ceil(col_max))
            edges = np.arange(low - 0.5, high + 1.5, 1.0)
        elif col_min == col_max:
            col_min -= 0.5
            col_max += 0.5
            edges = np.linspace(col_min, col_max, n_bins + 1)
        else:
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
