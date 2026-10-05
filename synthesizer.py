"""
synthesizer.py - Synthetic data generators for CrossSynth experiments.

Generators (all share .fit() / .generate() interface):
    1. CrossSynthGenerator      - Our method: local diffusion conditioned on
                                 aggregated cross-party marginals
    2. IndependentGenerator    - Each party trains TabDDPM independently
    3. CentralizedGenerator    - Pool all party data, train one TabDDPM (upper bound)
    4. PrivBayesGenerator      - Simplified PrivBayes: greedy Bayesian network with DP
    5. CTGANGenerator          - CTGAN wrapper (graceful skip if not installed)

The diffusion model reuses the architecture from paper2 (3-layer MLP, 256 hidden,
sinusoidal time embedding, 1000 timesteps) with an added marginal conditioning
vector concatenated to the MLP input.

Designed to run on CPU (32 GB RAM).
"""

import math
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from marginals import (
    compute_1way_marginals_shared,
    compute_2way_marginals_shared,
    compute_1way_counts_shared,
    compute_2way_counts_shared,
    compute_shared_bin_edges,
    column_domain_sizes,
    select_2way_pairs,
    add_dp_noise_1way,
    add_dp_noise_2way,
    aggregate_marginals_1way,
    aggregate_marginals_2way,
    add_dp_noise_count_queries,
    aggregate_count_queries,
    maybe_round_count_hists,
    scale_like_encoded_marginal,
    encode_marginals,
)


# ===================================================================
# Diffusion model components (reused from paper2 with conditioning)
# ===================================================================

def linear_beta_schedule(n_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linearly-spaced beta schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


class SinusoidalEmbedding(nn.Module):
    """Maps scalar timestep t -> vector of dimension `dim` via sinusoidal encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionedDenoisingMLP(nn.Module):
    """
    MLP that predicts noise eps_theta(x_t, t, c).

    Architecture:
        [x_t || time_emb || cond_vec] -> Linear -> ReLU -> Dropout
                                       -> Linear -> ReLU -> Dropout
                                       -> Linear -> ReLU -> Dropout
                                       -> Linear -> output (same dim as x_t)

    Parameters
    ----------
    input_dim : int
        Dimension of x_t (number of features).
    time_dim : int
        Dimension of sinusoidal time embedding.
    cond_dim : int
        Dimension of the conditioning vector (0 = unconditional).
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_dim, time_dim, cond_dim=0,
                 hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.cond_dim = cond_dim

        # Marginal projection: enrich conditioning vector via small MLP
        if cond_dim > 0:
            self.marginal_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            proj_dim = hidden_dim
        else:
            self.marginal_proj = None
            proj_dim = 0

        layers = []
        in_features = input_dim + time_dim + proj_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, cond=None):
        """
        Parameters
        ----------
        x_t : (B, input_dim)
        t   : (B,)
        cond: (B, cond_dim) or None
        """
        t_emb = self.time_embed(t)
        parts = [x_t, t_emb]
        if cond is not None and self.cond_dim > 0:
            cond_proj = self.marginal_proj(cond)
            parts.append(cond_proj)
        inp = torch.cat(parts, dim=-1)
        return self.net(inp)


def _bind_opacus_sample_rate(loader, sample_rate):
    """Make ``len(loader)`` the integer Opacus inverts to ``sample_rate``.

    Opacus 1.6 sets ``sample_rate = 1/len(data_loader)`` inside
    ``make_private_with_epsilon`` and also forwards unknown keywords into
    ``get_noise_multiplier``, which already binds ``sample_rate``. A second
    keyword raises ``TypeError``. ``len`` is looked up on the class, so the
    reported length has to live there. Rates above one are not Poisson
    probabilities; they are capped at one.
    """
    rate = float(sample_rate)
    if rate <= 0.0:
        raise ValueError("sample_rate q_k=B/n_k must be positive.")
    if rate > 1.0:
        rate = 1.0
    reported = max(1, int(round(1.0 / rate)))

    class _QLenLoader(loader.__class__):
        def __len__(self):
            return reported

    loader.__class__ = _QLenLoader
    return loader


def _opacus_privacy_engine():
    """Return an Opacus engine that accepts ``sample_rate=q_k``.

    The import stays inside this function so a missing Opacus install does
    not block count release. The engine is Opacus; the subclass only places
    ``q_k`` where ``make_private_with_epsilon`` will read it.
    """
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise ImportError(
            "Private diffusion training requires opacus. "
            "Install the dependencies from requirements.txt."
        ) from exc

    class _SampleRatePrivacyEngine(PrivacyEngine):
        def make_private_with_epsilon(self, *args, sample_rate=None, **kwargs):
            if sample_rate is not None:
                kwargs["data_loader"] = _bind_opacus_sample_rate(
                    kwargs["data_loader"], sample_rate
                )
            return super().make_private_with_epsilon(*args, **kwargs)

    return _SampleRatePrivacyEngine()


class TabularDiffusion(nn.Module):
    """
    Tabular diffusion model (simplified TabDDPM) with optional conditioning.

    Parameters
    ----------
    input_dim : int
        Number of features per sample.
    cond_dim : int
        Conditioning vector dimension (0 = unconditional).
    hidden_dim : int
        MLP hidden width (default 256).
    n_layers : int
        Number of hidden layers (default 3).
    n_timesteps : int
        Diffusion steps T (default 1000).
    """

    def __init__(self, input_dim, cond_dim=0, hidden_dim=256,
                 n_layers=3, n_timesteps=1000):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.n_timesteps = n_timesteps

        # Noise schedule
        betas = linear_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # Denoising network
        time_dim = min(128, hidden_dim)
        self.denoiser = ConditionedDenoisingMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    def compute_loss(self, x_0, cond=None):
        """DDPM training loss: E[||eps - eps_theta(x_t, t, c)||^2]."""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        predicted = self.denoiser(x_t, t, cond)
        loss = (noise - predicted).pow(2).mean()
        return loss

    def forward(self, x_0, cond=None):
        """Return the scalar denoising loss for ordinary or private training."""
        return self.compute_loss(x_0, cond)

    def train_model(self, X_train, cond_vec=None, epochs=50,
                    batch_size=256, lr=1e-3, verbose=True,
                    dp_epsilon=None, dp_delta=1e-5,
                    max_grad_norm=1.0):
        """
        Train the diffusion model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, d)
        cond_vec : np.ndarray, shape (cond_dim,) or (n, cond_dim) or None
            A one-dimensional vector is broadcast to every row. That is the
            CrossSynth protocol input. A matrix remains accepted when a
            caller has already built one condition per row.
        epochs : int
        batch_size : int
        lr : float
        verbose : bool

        Returns
        -------
        losses : list[float]
        """
        self.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        # One vector is broadcast. A matrix is a per-row condition.
        cond_tensor = None
        if cond_vec is not None and self.cond_dim > 0:
            cond_tensor = torch.tensor(cond_vec, dtype=torch.float32)
            if cond_tensor.ndim == 1:
                cond_tensor = cond_tensor.unsqueeze(0).expand(
                    X_tensor.shape[0], -1
                )
            elif cond_tensor.ndim != 2 or cond_tensor.shape[0] != X_tensor.shape[0]:
                raise ValueError(
                    "cond_vec must have shape (cond_dim,) or (n, cond_dim)."
                )

        if cond_tensor is not None:
            dataset = TensorDataset(X_tensor, cond_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_model = self
        privacy_engine = None
        if dp_epsilon is not None:
            n_k = int(X_tensor.shape[0])
            q_k = float(batch_size) / float(n_k)
            privacy_engine = _opacus_privacy_engine()
            training_model, optimizer, loader = (
                privacy_engine.make_private_with_epsilon(
                    module=self,
                    optimizer=optimizer,
                    data_loader=loader,
                    epochs=epochs,
                    target_epsilon=float(dp_epsilon),
                    target_delta=float(dp_delta),
                    max_grad_norm=float(max_grad_norm),
                    sample_rate=q_k,
                )
            )
        losses = []

        epoch_iter = tqdm(range(epochs), desc="TabDDPM training",
                          disable=not verbose, ascii=True)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if cond_tensor is not None:
                    x_batch, c_batch = batch
                else:
                    x_batch = batch[0]
                    c_batch = None

                optimizer.zero_grad()
                loss = training_model(x_batch, c_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        self.privacy_epsilon = None
        self.privacy_delta = None
        if privacy_engine is not None:
            self.privacy_epsilon = float(privacy_engine.get_epsilon(dp_delta))
            self.privacy_delta = float(dp_delta)
        return losses

    @torch.no_grad()
    def sample(self, n_samples, cond_vec=None, verbose=False):
        """
        Generate synthetic rows via DDPM reverse process.

        Parameters
        ----------
        n_samples : int
        cond_vec : np.ndarray, shape (cond_dim,) or (n_samples, cond_dim) or None
        verbose : bool

        Returns
        -------
        X_synthetic : np.ndarray, shape (n_samples, input_dim)
        """
        self.eval()
        x = torch.randn(n_samples, self.input_dim)

        # Prepare conditioning
        cond = None
        if cond_vec is not None and self.cond_dim > 0:
            cond = torch.tensor(cond_vec, dtype=torch.float32)
            if cond.ndim == 1:
                cond = cond.unsqueeze(0).expand(n_samples, -1)
            elif cond.ndim != 2 or cond.shape[0] != n_samples:
                raise ValueError(
                    "cond_vec must have shape (cond_dim,) or "
                    "(n_samples, cond_dim)."
                )

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Sampling", disable=not verbose, ascii=True)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            predicted_noise = self.denoiser(x, t, cond)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        return x.numpy()


# ===================================================================
# Generator base class
# ===================================================================

class BaseGenerator:
    """
    Abstract base for all synthetic data generators.

    Interface
    ---------
    .fit(parties, epsilon, **kwargs)
        Train the generator given K parties and a privacy budget.
    .generate(n_samples)
        Generate n_samples synthetic rows as np.ndarray.
    """

    def __init__(self, name="BaseGenerator"):
        self.name = name

    def fit(self, parties, epsilon=1.0, **kwargs):
        raise NotImplementedError

    def generate(self, n_samples):
        raise NotImplementedError


def _conditioning_vector(encoded, cond_mode, rng_seed=42):
    """Same-length conditioning control for a CrossSynth network.

    ``marginal`` returns the encoded aggregate. ``constant`` is a vector of
    ones on that vector's scale. ``random`` is a seeded Gaussian, then the
    same scale. ``none`` is a zero vector of that length. It does not change
    ``cond_dim``.
    """
    encoded = np.asarray(encoded, dtype=np.float32)
    dim = int(encoded.shape[0])
    if cond_mode == "marginal":
        return encoded
    if cond_mode == "none":
        return np.zeros(dim, dtype=np.float32)
    if cond_mode == "constant":
        raw = np.ones(dim, dtype=np.float32)
    elif cond_mode == "random":
        rng = np.random.RandomState(rng_seed)
        raw = rng.normal(size=dim).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown cond_mode '{cond_mode}'. "
            "Choose from: marginal, constant, random, none."
        )
    return scale_like_encoded_marginal(raw, encoded)


def _controlled_marginal_targets(marginals_1way, marginals_2way,
                                 cond_mode, rng_seed=42):
    """Return real or matched-shape control workloads for calibration."""
    if cond_mode == "none":
        return None, None
    rng = np.random.RandomState(rng_seed)

    def _transform(items):
        transformed = []
        for item in items or []:
            copied = dict(item)
            shape = np.asarray(item["hist"]).shape
            if cond_mode == "marginal":
                target = np.asarray(item["hist"], dtype=np.float64).copy()
            elif cond_mode == "constant":
                target = np.full(shape, 1.0 / np.prod(shape))
            elif cond_mode == "random":
                draw = rng.gamma(shape=1.0, scale=1.0, size=shape)
                target = draw / draw.sum()
            else:
                raise ValueError(f"Unknown cond_mode '{cond_mode}'.")
            copied["hist"] = target
            transformed.append(copied)
        return transformed

    return _transform(marginals_1way), _transform(marginals_2way)


def _pad_condition(parts, n_rows, max_dim):
    """Concatenate compact condition fields and pad/truncate deterministically."""
    if parts:
        matrix = np.column_stack(parts).astype(np.float32)
    else:
        matrix = np.zeros((n_rows, 0), dtype=np.float32)
    if matrix.shape[1] > max_dim:
        return matrix[:, :max_dim]
    if matrix.shape[1] < max_dim:
        padding = np.zeros(
            (n_rows, max_dim - matrix.shape[1]), dtype=np.float32
        )
        matrix = np.column_stack([matrix, padding])
    return matrix


def _row_marginal_conditions(X, marginals_1way, marginals_2way,
                             max_dim=256):
    """Encode every row's cells under a released marginal workload.

    A one-way query contributes a normalized bin index and the released
    probability of that cell. A two-way query contributes two normalized bin
    indices and the released joint-cell probability. These anchors vary by
    row, so they cannot be folded into the denoiser bias.
    """
    X = np.asarray(X)
    parts = []
    for item in marginals_1way or []:
        edges = np.asarray(item["edges"])
        hist = np.asarray(item["hist"], dtype=np.float64).reshape(-1)
        col = int(item["col"])
        bins = np.clip(
            np.searchsorted(edges, X[:, col], side="right") - 1,
            0, hist.size - 1,
        )
        parts.extend([bins / max(hist.size - 1, 1), hist[bins]])

    for item in marginals_2way or []:
        ci, cj = (int(v) for v in item["cols"])
        ex = np.asarray(item["edges_x"])
        ey = np.asarray(item["edges_y"])
        hist = np.asarray(item["hist"], dtype=np.float64)
        bi = np.clip(
            np.searchsorted(ex, X[:, ci], side="right") - 1,
            0, hist.shape[0] - 1,
        )
        bj = np.clip(
            np.searchsorted(ey, X[:, cj], side="right") - 1,
            0, hist.shape[1] - 1,
        )
        parts.extend([
            bi / max(hist.shape[0] - 1, 1),
            bj / max(hist.shape[1] - 1, 1),
            hist[bi, bj],
        ])
    return _pad_condition(parts, X.shape[0], max_dim)


def _sample_marginal_conditions(marginals_1way, marginals_2way, n_samples,
                                max_dim=256, rng_seed=42):
    """Sample row-varying anchors solely from the released workload."""
    rng = np.random.RandomState(rng_seed)
    parts = []
    for item in marginals_1way or []:
        hist = np.asarray(item["hist"], dtype=np.float64).reshape(-1)
        probs = np.maximum(hist, 0.0)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(
            hist.size, 1.0 / hist.size
        )
        bins = rng.choice(hist.size, size=n_samples, p=probs)
        parts.extend([bins / max(hist.size - 1, 1), probs[bins]])

    for item in marginals_2way or []:
        hist = np.asarray(item["hist"], dtype=np.float64)
        probs = np.maximum(hist.reshape(-1), 0.0)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(
            probs.size, 1.0 / probs.size
        )
        flat = rng.choice(probs.size, size=n_samples, p=probs)
        bi, bj = np.unravel_index(flat, hist.shape)
        parts.extend([
            bi / max(hist.shape[0] - 1, 1),
            bj / max(hist.shape[1] - 1, 1),
            probs[flat],
        ])
    return _pad_condition(parts, n_samples, max_dim)


def _condition_control(matrix, cond_mode, rng_seed=42):
    """Create a shape-matched control for an informative anchor matrix."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if cond_mode == "marginal":
        return matrix
    if cond_mode == "none":
        return np.zeros_like(matrix)
    if cond_mode == "constant":
        return np.full_like(matrix, 0.5)
    if cond_mode == "random":
        rng = np.random.RandomState(rng_seed)
        return rng.uniform(0.0, 1.0, size=matrix.shape).astype(np.float32)
    raise ValueError(f"Unknown cond_mode '{cond_mode}'.")


def _sample_released_discrete_column(marginals_1way, column, n_samples,
                                     rng_seed=42):
    """Sample a discrete column using only its released one-way marginal."""
    match = next(
        (item for item in (marginals_1way or [])
         if int(item["col"]) == int(column)),
        None,
    )
    if match is None:
        raise ValueError(f"No released one-way marginal for column {column}.")
    hist = np.maximum(
        np.asarray(match["hist"], dtype=np.float64).reshape(-1), 0.0
    )
    probs = hist / hist.sum() if hist.sum() > 0 else np.full(
        hist.size, 1.0 / hist.size
    )
    rng = np.random.RandomState(rng_seed)
    bins = rng.choice(hist.size, size=n_samples, p=probs)
    edges = np.asarray(match["edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.rint(centers[bins]).astype(np.float32)


def marginal_reweight_resample(X_candidates, target_1way, target_2way,
                               n_samples, passes=5, damping=0.5,
                               rng_seed=42):
    """Select a synthetic shard that matches a released marginal workload.

    This is deterministic post-processing up to the seeded final draw. It uses
    iterative proportional reweighting over one- and two-way histogram cells,
    with damped and clipped ratios to avoid a single sparse cell collapsing
    the effective sample size.
    """
    X_candidates = np.asarray(X_candidates)
    n_candidates = X_candidates.shape[0]
    if n_candidates == 0:
        raise ValueError("Candidate pool is empty.")
    if not target_1way and not target_2way:
        rng = np.random.RandomState(rng_seed)
        index = rng.choice(n_candidates, n_samples, replace=True)
        return X_candidates[index], float(n_candidates)

    weights = np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)

    def _bin_1way(item):
        edges = np.asarray(item["edges"])
        column = int(item["col"])
        return np.clip(
            np.searchsorted(edges, X_candidates[:, column], side="right") - 1,
            0,
            len(edges) - 2,
        )

    one_cache = [(_bin_1way(item), np.asarray(item["hist"]).reshape(-1))
                 for item in (target_1way or [])]
    two_cache = []
    for item in target_2way or []:
        ci, cj = item["cols"]
        ex = np.asarray(item["edges_x"])
        ey = np.asarray(item["edges_y"])
        bi = np.clip(
            np.searchsorted(ex, X_candidates[:, ci], side="right") - 1,
            0, len(ex) - 2,
        )
        bj = np.clip(
            np.searchsorted(ey, X_candidates[:, cj], side="right") - 1,
            0, len(ey) - 2,
        )
        flat = bi * (len(ey) - 1) + bj
        two_cache.append((flat, np.asarray(item["hist"]).reshape(-1)))

    for _ in range(max(int(passes), 1)):
        for cell_index, target in one_cache + two_cache:
            current = np.bincount(
                cell_index, weights=weights, minlength=target.size
            ).astype(np.float64)
            ratio = (target + 1e-8) / (current + 1e-8)
            ratio = np.clip(ratio, 0.1, 10.0) ** float(damping)
            weights *= ratio[cell_index]
            total = weights.sum()
            if not np.isfinite(total) or total <= 0:
                weights.fill(1.0 / n_candidates)
            else:
                weights /= total

    effective_sample_size = float(1.0 / np.sum(weights ** 2))
    rng = np.random.RandomState(rng_seed)
    chosen = rng.choice(n_candidates, n_samples, replace=True, p=weights)
    return X_candidates[chosen], effective_sample_size


# ===================================================================
# 1. CrossSynth Generator (our method)
# ===================================================================

class CrossSynthGenerator(BaseGenerator):
    """
    CrossSynth: each party trains a local diffusion model conditioned on
    securely aggregated cross-party marginals.

    Steps:
        1. All parties agree on shared bin edges.
        2. Each party releases one concatenated count workload. Gaussian
           noise uses replace-one sensitivity sqrt(2 * |Q|).
        3. Counts are rounded, then summed in-process (modular shares when
           secure aggregation is on) and turned into one probability table.
        4. That table is encoded as one fixed vector. constant, random, and
           none replace it with one scaled ones vector, one scaled Gaussian,
           or zeros, all of the same length.
        5. Each party trains a local TabDDPM on that same vector. Private
           training spends the remaining budget through Opacus at rate B/n_k.
        6. Sampling uses the same vector at every step. No raw row is
           consulted.
    """

    def __init__(self, hidden_dim=256, n_layers=3, n_timesteps=1000,
                 cond_dim=256, n_bins=20):
        super().__init__(name="CrossSynth")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.cond_dim = cond_dim
        self.n_bins = n_bins
        self.models = []
        self.cond_vec = None
        self.party_sizes = []
        self.data_mean = None
        self.data_std = None
        self.data_means = []
        self.data_stds = []
        self.release_metadata = {}
        self.target_marginals_1way = None
        self.target_marginals_2way = None
        self.training_conditions = []

    def fit(self, parties, epsilon=1.0, epochs=50, batch_size=256,
            lr=1e-3, verbose=True, delta=1e-5,
            cond_mode="marginal", M=50, selection="domain_size",
            eps_marginal_frac=0.5, round_counts=False,
            marginal_mode="legacy_probability", secure_aggregation=False,
            modulus=2_147_483_647, private_training=False,
            delta_marginal_frac=0.5, max_grad_norm=1.0,
            normalization="global", calibrate_output=False,
            calibration_oversample=2.0, calibration_passes=5,
            calibration_include_2way=True, calibration_damping=0.5,
            categorical_indices=None, **kwargs):
        """
        Train CrossSynth across all parties.

        Privacy budget allocation:
            - eps_marginal_frac * epsilon for the count workload.
              This argument still defaults to 0.5 for older callers.
              The paper CLI passes 0.1.
            - delta_marginal_frac * delta for that same workload.
              This argument still defaults to 0.5 for older callers.
              The paper CLI passes 0.1.
            - the complement is the private training budget when
              private_training is true

        cond_mode:
            marginal — one encoded aggregate (default)
            constant — one scaled vector of ones, same length
            random   — one scaled Gaussian draw, same length
            none     — one zero vector, same length
        """
        K = len(parties)
        d = parties[0]["X"].shape[1]
        if cond_mode not in ("marginal", "constant", "random", "none"):
            raise ValueError(
                f"Unknown cond_mode '{cond_mode}'. "
                "Choose from: marginal, constant, random, none."
            )
        if selection not in ("domain_size", "random"):
            raise ValueError(
                f"Unknown selection '{selection}'. "
                "Choose from: domain_size, random."
            )
        if marginal_mode not in ("legacy_probability", "count"):
            raise ValueError(
                "marginal_mode must be 'legacy_probability' or 'count'."
            )
        if not 0.0 < eps_marginal_frac < 1.0:
            raise ValueError("eps_marginal_frac must lie strictly between 0 and 1.")
        if not 0.0 < delta_marginal_frac < 1.0:
            raise ValueError("delta_marginal_frac must lie strictly between 0 and 1.")
        if normalization not in ("global", "local"):
            raise ValueError("normalization must be 'global' or 'local'.")
        if secure_aggregation and not round_counts:
            raise ValueError(
                "Modular secure aggregation requires round_counts=True."
            )
        # The paper CLI passes 0.1 on the count release and 0.9 on training.
        eps_marginals = epsilon * eps_marginal_frac
        eps_training = epsilon * (1.0 - eps_marginal_frac)
        delta_marginals = delta * delta_marginal_frac
        delta_training = delta * (1.0 - delta_marginal_frac)

        print(f"[CrossSynth] Training with K={K} parties, eps={epsilon}, "
              f"d={d}, epochs={epochs}, cond_mode={cond_mode}, "
              f"M={M}, selection={selection}")

        # Step 1: Agree on shared bin edges
        self.categorical_indices = sorted(set(categorical_indices or []))
        shared_edges = compute_shared_bin_edges(
            parties, n_bins=self.n_bins,
            categorical_indices=self.categorical_indices,
        )
        self.shared_edges = shared_edges
        domain_sizes = column_domain_sizes(shared_edges)
        self.selected_pairs = select_2way_pairs(
            d, M=M, selection=selection, domain_sizes=domain_sizes,
        )
        self.cond_mode = cond_mode

        all_noisy_marginals = []
        all_noisy_2way = []
        release_sigmas = []
        release_sensitivity = None
        if marginal_mode == "legacy_probability":
            # Historical probability-space path retained for reproducibility.
            # Finish every one-way draw before any two-way draw so its random
            # sequence remains compatible with the earlier runner.
            for k in range(K):
                m1 = compute_1way_marginals_shared(
                    parties[k]["X"], shared_edges
                )
                all_noisy_marginals.append(add_dp_noise_1way(
                    m1, n_samples=parties[k]["X"].shape[0],
                    epsilon=eps_marginals, delta=delta,
                ))
            for k in range(K):
                m2 = compute_2way_marginals_shared(
                    parties[k]["X"], self.selected_pairs, shared_edges
                )
                all_noisy_2way.append(add_dp_noise_2way(
                    m2, n_samples=parties[k]["X"].shape[0],
                    epsilon=eps_marginals, delta=delta,
                ) if m2 else [])
            if round_counts:
                all_noisy_marginals = [
                    maybe_round_count_hists(m, True)
                    for m in all_noisy_marginals
                ]
                all_noisy_2way = [
                    maybe_round_count_hists(m, True)
                    for m in all_noisy_2way
                ]
            agg_marginals = aggregate_marginals_1way(all_noisy_marginals)
            agg_2way = (
                aggregate_marginals_2way(all_noisy_2way)
                if self.selected_pairs else None
            )
        else:
            # Manuscript path: release a single concatenated count workload.
            for k in range(K):
                m1 = compute_1way_counts_shared(
                    parties[k]["X"], shared_edges
                )
                m2 = compute_2way_counts_shared(
                    parties[k]["X"], self.selected_pairs, shared_edges
                )
                noisy_1, noisy_2, sigma, sensitivity = (
                    add_dp_noise_count_queries(
                        m1, m2, epsilon=eps_marginals,
                        delta=delta_marginals,
                    )
                )
                all_noisy_marginals.append(
                    maybe_round_count_hists(noisy_1, round_counts)
                )
                all_noisy_2way.append(
                    maybe_round_count_hists(noisy_2, round_counts)
                )
                release_sigmas.append(sigma)
                release_sensitivity = sensitivity
            agg_marginals, agg_2way = aggregate_count_queries(
                all_noisy_marginals,
                all_noisy_2way,
                secure=secure_aggregation,
                modulus=modulus,
            )

        # One fixed vector. constant / random / none stay the same length
        # and do not vary by row.
        encoded = encode_marginals(
            agg_marginals, marginals_2way=agg_2way, max_dim=self.cond_dim
        )
        self.cond_vec = _conditioning_vector(encoded, cond_mode)
        self.target_marginals_1way, self.target_marginals_2way = (
            _controlled_marginal_targets(
                agg_marginals, agg_2way, cond_mode=cond_mode,
            )
        )
        self.calibrate_output = bool(calibrate_output)
        self.calibration_oversample = max(float(calibration_oversample), 1.0)
        self.calibration_passes = max(int(calibration_passes), 1)
        self.calibration_include_2way = bool(calibration_include_2way)
        self.calibration_damping = float(calibration_damping)
        self.training_conditions = [self.cond_vec for _ in range(K)]

        self.normalization = normalization
        self.data_means = []
        self.data_stds = []
        if normalization == "global":
            X_all_parts = np.concatenate([p["X"] for p in parties], axis=0)
            self.data_mean = X_all_parts.mean(axis=0).astype(np.float32)
            self.data_std = X_all_parts.std(axis=0).astype(np.float32)
            self.data_std[self.data_std < 1e-8] = 1.0
            self.data_means = [self.data_mean for _ in parties]
            self.data_stds = [self.data_std for _ in parties]
        else:
            for party in parties:
                mean = party["X"].mean(axis=0).astype(np.float32)
                std = party["X"].std(axis=0).astype(np.float32)
                std[std < 1e-8] = 1.0
                self.data_means.append(mean)
                self.data_stds.append(std)
            self.data_mean = None
            self.data_std = None

        n_bins_released = sum(item["hist"].size for item in agg_marginals)
        n_bins_released += sum(item["hist"].size for item in (agg_2way or []))
        self.release_metadata = {
            "marginal_mode": marginal_mode,
            "query_count": int(d + len(self.selected_pairs)),
            "released_bins": int(n_bins_released),
            "selected_pairs": [list(pair) for pair in self.selected_pairs],
            "epsilon_marginals": float(eps_marginals),
            "epsilon_training": float(eps_training),
            "delta_marginals": float(delta_marginals),
            "delta_training": float(delta_training),
            "count_sensitivity": release_sensitivity,
            "count_sigma": release_sigmas[0] if release_sigmas else None,
            "rounded": bool(round_counts),
            "secure_aggregation": bool(secure_aggregation),
            "modulus": int(modulus) if secure_aggregation else None,
            "normalization": normalization,
            "calibrate_output": bool(calibrate_output),
            "calibration_oversample": float(self.calibration_oversample),
            "calibration_passes": int(self.calibration_passes),
            "calibration_include_2way": bool(self.calibration_include_2way),
            "calibration_damping": float(self.calibration_damping),
        }

        # Step 5: Each party trains a local conditioned diffusion model.
        # The paper profile applies the complementary budget through Opacus.
        self.models = []
        self.party_sizes = []

        for k in range(K):
            print(f"  Party {k}/{K}: {parties[k]['X'].shape[0]} samples")
            X_norm = (
                parties[k]["X"] - self.data_means[k]
            ) / self.data_stds[k]
            model = TabularDiffusion(
                input_dim=d,
                cond_dim=self.cond_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                n_timesteps=self.n_timesteps,
            )
            model.train_model(
                X_norm,
                cond_vec=self.cond_vec,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=verbose,
                dp_epsilon=eps_training if private_training else None,
                dp_delta=delta_training,
                max_grad_norm=max_grad_norm,
            )
            self.models.append(model)
            self.party_sizes.append(parties[k]["X"].shape[0])

        self.release_metadata["private_training"] = bool(private_training)
        self.release_metadata["achieved_training_epsilons"] = [
            getattr(model, "privacy_epsilon", None) for model in self.models
        ]

        print(f"[CrossSynth] Training complete.")

    def generate(self, n_samples):
        """
        Generate synthetic data by sampling proportionally from each
        party's model, then union.
        """
        if not self.models:
            raise RuntimeError("Must call .fit() before .generate()")

        total = sum(self.party_sizes)
        all_samples = []

        for k, model in enumerate(self.models):
            # Proportional allocation
            n_k = int(round(n_samples * self.party_sizes[k] / total))
            if n_k == 0:
                n_k = 1
            n_candidates = (
                int(math.ceil(n_k * self.calibration_oversample))
                if self.calibrate_output else n_k
            )
            samples = model.sample(
                n_candidates, cond_vec=self.cond_vec, verbose=False
            )
            samples = samples * self.data_stds[k] + self.data_means[k]
            for col in self.categorical_indices:
                edges = self.shared_edges[col]
                lower = float(edges[0] + 0.5)
                upper = float(edges[-1] - 0.5)
                samples[:, col] = np.clip(
                    np.rint(samples[:, col]), lower, upper
                )
            if self.calibrate_output:
                samples, ess = marginal_reweight_resample(
                    samples,
                    self.target_marginals_1way,
                    (self.target_marginals_2way
                     if self.calibration_include_2way else None),
                    n_samples=n_k,
                    passes=self.calibration_passes,
                    damping=self.calibration_damping,
                    rng_seed=42 + k,
                )
                self.release_metadata.setdefault(
                    "calibration_effective_sample_sizes", []
                ).append(float(ess))
            all_samples.append(samples)

        result = np.concatenate(all_samples, axis=0)

        # Trim or pad to exact n_samples
        if result.shape[0] > n_samples:
            result = result[:n_samples]
        elif result.shape[0] < n_samples:
            extra = n_samples - result.shape[0]
            idx = np.random.choice(result.shape[0], extra, replace=True)
            result = np.concatenate([result, result[idx]], axis=0)

        return result


# ===================================================================
# 2. Federated parameter averaging (communication baseline)
# ===================================================================

class FedAvgGenerator(BaseGenerator):
    """Federated diffusion training with optional delta compression.

    This baseline is intentionally separate from CrossSynth: it communicates
    model deltas for several rounds. ``int8`` applies symmetric per-tensor
    quantisation before aggregation; ``none`` sends dense float32 deltas.
    """

    def __init__(self, hidden_dim=256, n_layers=3, n_timesteps=1000,
                 rounds=20, local_epochs=1, compression="none"):
        super().__init__(name=f"FedAvg-{compression}")
        if compression not in ("none", "int8"):
            raise ValueError("compression must be 'none' or 'int8'.")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.rounds = int(rounds)
        self.local_epochs = int(local_epochs)
        self.compression = compression
        self.model = None
        self.communication_metadata = {}

    @staticmethod
    def _compress(delta, compression):
        if compression == "none":
            return delta, delta.numel() * 4
        max_abs = float(delta.abs().max())
        if max_abs == 0.0:
            return torch.zeros_like(delta), delta.numel() + 4
        scale = max_abs / 127.0
        quantized = torch.clamp(torch.round(delta / scale), -127, 127)
        return quantized * scale, delta.numel() + 4

    def fit(self, parties, epsilon=1.0, epochs=None, batch_size=256,
            lr=1e-3, verbose=True, **kwargs):
        del epsilon
        K = len(parties)
        d = parties[0]["X"].shape[1]
        if epochs is not None:
            total_local_epochs = max(int(epochs), 1)
            rounds = min(self.rounds, total_local_epochs)
            local_epochs = max(total_local_epochs // rounds, 1)
        else:
            rounds = self.rounds
            local_epochs = self.local_epochs

        pooled = np.concatenate([party["X"] for party in parties], axis=0)
        self.data_mean = pooled.mean(axis=0).astype(np.float32)
        self.data_std = pooled.std(axis=0).astype(np.float32)
        self.data_std[self.data_std < 1e-8] = 1.0
        self.model = TabularDiffusion(
            input_dim=d,
            cond_dim=0,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        transmitted_per_party = 0
        party_weights = np.asarray(
            [party["X"].shape[0] for party in parties], dtype=np.float64
        )
        party_weights /= party_weights.sum()

        for round_idx in range(rounds):
            base_state = copy.deepcopy(self.model.state_dict())
            parameter_deltas = []
            round_bytes = []
            for party in parties:
                local_model = TabularDiffusion(
                    input_dim=d,
                    cond_dim=0,
                    hidden_dim=self.hidden_dim,
                    n_layers=self.n_layers,
                    n_timesteps=self.n_timesteps,
                )
                local_model.load_state_dict(base_state)
                normalized = (party["X"] - self.data_mean) / self.data_std
                local_model.train_model(
                    normalized,
                    epochs=local_epochs,
                    batch_size=min(batch_size, normalized.shape[0]),
                    lr=lr,
                    verbose=False,
                )
                deltas = {}
                bytes_sent = 0
                for name, parameter in local_model.named_parameters():
                    delta = parameter.detach() - base_state[name]
                    reconstructed, n_bytes = self._compress(
                        delta, self.compression
                    )
                    deltas[name] = reconstructed
                    bytes_sent += n_bytes
                parameter_deltas.append(deltas)
                round_bytes.append(bytes_sent)

            updated = copy.deepcopy(base_state)
            for name, _ in self.model.named_parameters():
                average_delta = sum(
                    float(party_weights[k]) * parameter_deltas[k][name]
                    for k in range(K)
                )
                updated[name] = base_state[name] + average_delta
            self.model.load_state_dict(updated)
            transmitted_per_party += int(np.mean(round_bytes))
            if verbose:
                print(f"[FedAvg-{self.compression}] round "
                      f"{round_idx + 1}/{rounds}")

        parameter_count = sum(p.numel() for p in self.model.parameters())
        self.communication_metadata = {
            "compression": self.compression,
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "parameter_count": int(parameter_count),
            "bytes_per_party_total_upload": int(transmitted_per_party),
            "bytes_network_total_upload": int(transmitted_per_party * K),
        }

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("Must call .fit() before .generate()")
        samples = self.model.sample(n_samples, verbose=False)
        return samples * self.data_std + self.data_mean


# ===================================================================
# 3. Independent Generator (no cross-party info)
# ===================================================================

class IndependentGenerator(BaseGenerator):
    """
    Each party trains TabDDPM independently - no cross-party information.
    This is the baseline showing degradation without federation.
    """

    def __init__(self, hidden_dim=256, n_layers=3, n_timesteps=1000):
        super().__init__(name="Independent")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.models = []
        self.party_sizes = []
        self.data_mean = None
        self.data_std = None
        self.data_means = []
        self.data_stds = []

    def fit(self, parties, epsilon=1.0, epochs=50, batch_size=256,
            lr=1e-3, verbose=True, private_training=False,
            delta=1e-5, max_grad_norm=1.0,
            normalization="global", **kwargs):
        K = len(parties)
        d = parties[0]["X"].shape[1]
        print(f"[Independent] Training with K={K} parties, eps={epsilon}, "
              f"d={d}, epochs={epochs}")

        self.data_means = []
        self.data_stds = []
        if normalization == "global":
            X_all_parts = np.concatenate([p["X"] for p in parties], axis=0)
            self.data_mean = X_all_parts.mean(axis=0).astype(np.float32)
            self.data_std = X_all_parts.std(axis=0).astype(np.float32)
            self.data_std[self.data_std < 1e-8] = 1.0
            self.data_means = [self.data_mean for _ in parties]
            self.data_stds = [self.data_std for _ in parties]
        elif normalization == "local":
            for party in parties:
                mean = party["X"].mean(axis=0).astype(np.float32)
                std = party["X"].std(axis=0).astype(np.float32)
                std[std < 1e-8] = 1.0
                self.data_means.append(mean)
                self.data_stds.append(std)
        else:
            raise ValueError("normalization must be 'global' or 'local'.")

        self.models = []
        self.party_sizes = []

        for k in range(K):
            print(f"  Party {k}/{K}: {parties[k]['X'].shape[0]} samples")
            X_norm = (
                parties[k]["X"] - self.data_means[k]
            ) / self.data_stds[k]
            model = TabularDiffusion(
                input_dim=d,
                cond_dim=0,  # No conditioning
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                n_timesteps=self.n_timesteps,
            )
            model.train_model(
                X_norm,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=verbose,
                dp_epsilon=epsilon if private_training else None,
                dp_delta=delta,
                max_grad_norm=max_grad_norm,
            )
            self.models.append(model)
            self.party_sizes.append(parties[k]["X"].shape[0])

        print(f"[Independent] Training complete.")

    def generate(self, n_samples):
        if not self.models:
            raise RuntimeError("Must call .fit() before .generate()")

        total = sum(self.party_sizes)
        all_samples = []

        for k, model in enumerate(self.models):
            n_k = int(round(n_samples * self.party_sizes[k] / total))
            if n_k == 0:
                n_k = 1
            samples = model.sample(n_k, verbose=False)
            samples = samples * self.data_stds[k] + self.data_means[k]
            all_samples.append(samples)

        result = np.concatenate(all_samples, axis=0)
        if result.shape[0] > n_samples:
            result = result[:n_samples]
        elif result.shape[0] < n_samples:
            extra = n_samples - result.shape[0]
            idx = np.random.choice(result.shape[0], extra, replace=True)
            result = np.concatenate([result, result[idx]], axis=0)

        return result


# ===================================================================
# 3. Centralized Generator (upper bound, violates privacy)
# ===================================================================

class CentralizedGenerator(BaseGenerator):
    """
    Pool all party data and train a single TabDDPM.
    This is the upper bound - violates privacy by centralizing data.
    """

    def __init__(self, hidden_dim=256, n_layers=3, n_timesteps=1000):
        super().__init__(name="Centralized")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_timesteps = n_timesteps
        self.model = None
        self.data_mean = None
        self.data_std = None

    def fit(self, parties, epsilon=1.0, epochs=50, batch_size=256,
            lr=1e-3, verbose=True, **kwargs):
        # Pool all party data
        X_all = np.concatenate([p["X"] for p in parties], axis=0)
        d = X_all.shape[1]
        print(f"[Centralized] Training on pooled data: "
              f"{X_all.shape[0]} samples, d={d}, epochs={epochs}")

        # Normalize ALL features for diffusion training (zero mean, unit std)
        self.data_mean = X_all.mean(axis=0).astype(np.float32)
        self.data_std = X_all.std(axis=0).astype(np.float32)
        self.data_std[self.data_std < 1e-8] = 1.0
        X_norm = (X_all - self.data_mean) / self.data_std

        self.model = TabularDiffusion(
            input_dim=d,
            cond_dim=0,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_timesteps=self.n_timesteps,
        )
        self.model.train_model(
            X_norm,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        print(f"[Centralized] Training complete.")

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("Must call .fit() before .generate()")
        samples = self.model.sample(n_samples, verbose=False)
        samples = samples * self.data_std + self.data_mean
        return samples


# ===================================================================
# 4. PrivBayes Generator (simplified)
# ===================================================================

class PrivBayesGenerator(BaseGenerator):
    """
    Simplified PrivBayes: greedy Bayesian network with DP noise.

    Steps per party:
        1. Discretize all features into bins.
        2. Learn attribute ordering greedily by mutual information.
        3. For each attribute, learn a noisy conditional distribution
           given its parent(s) with DP.
        4. Sample by following the ordering.

    All party outputs are unioned for the final result.
    """

    def __init__(self, n_bins=15, max_parents=2):
        super().__init__(name="PrivBayes")
        self.n_bins = n_bins
        self.max_parents = max_parents
        self.party_models = []
        self.party_sizes = []

    def _discretize(self, X):
        """Discretize each column into bins. Returns bin indices and edges."""
        n, d = X.shape
        X_disc = np.zeros_like(X, dtype=int)
        all_edges = []

        for j in range(d):
            col = X[:, j]
            col_min, col_max = col.min(), col.max()
            if col_min == col_max:
                col_min -= 0.5
                col_max += 0.5
            edges = np.linspace(col_min - 1e-6, col_max + 1e-6,
                                self.n_bins + 1)
            X_disc[:, j] = np.clip(
                np.digitize(col, edges[1:-1]), 0, self.n_bins - 1
            )
            all_edges.append(edges)

        return X_disc, all_edges

    def _noisy_mutual_info(self, X_disc, col_a, col_b, n_samples, epsilon):
        """Compute noisy mutual information between two discretized columns."""
        # Joint distribution
        joint = np.zeros((self.n_bins, self.n_bins), dtype=np.float64)
        for i in range(n_samples):
            a, b = int(X_disc[i, col_a]), int(X_disc[i, col_b])
            a = min(a, self.n_bins - 1)
            b = min(b, self.n_bins - 1)
            joint[a, b] += 1

        # Add noise
        sensitivity = 2.0 / max(n_samples, 1)
        if epsilon > 0:
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
            joint += np.random.normal(0, sigma, joint.shape)

        joint = np.maximum(joint, 0)
        total = joint.sum()
        if total == 0:
            return 0.0
        joint = joint / total

        # Marginals
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)

        # MI
        mi = 0.0
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                if joint[i, j] > 1e-12 and p_a[i] > 1e-12 and p_b[j] > 1e-12:
                    mi += joint[i, j] * np.log(joint[i, j] / (p_a[i] * p_b[j]))

        return max(mi, 0.0)

    def _learn_ordering(self, X_disc, n_samples, epsilon):
        """Greedy attribute ordering by mutual information."""
        d = X_disc.shape[1]
        eps_per_pair = epsilon / max(d * (d - 1) / 2, 1)

        remaining = list(range(d))
        ordering = []
        parents = {}

        # First attribute: pick the one with highest variance
        variances = [X_disc[:, j].var() for j in remaining]
        first = remaining[int(np.argmax(variances))]
        ordering.append(first)
        parents[first] = []
        remaining.remove(first)

        while remaining:
            best_col = None
            best_score = -1
            best_parent = None

            for col in remaining:
                for p in ordering[-self.max_parents:]:
                    score = self._noisy_mutual_info(
                        X_disc, col, p, n_samples, eps_per_pair
                    )
                    if score > best_score:
                        best_score = score
                        best_col = col
                        best_parent = p

            if best_col is not None:
                ordering.append(best_col)
                # Assign parent(s)
                parent_list = [best_parent] if best_parent is not None else []
                parents[best_col] = parent_list
                remaining.remove(best_col)
            else:
                # Fallback: just add remaining
                col = remaining[0]
                ordering.append(col)
                parents[col] = []
                remaining.remove(col)

        return ordering, parents

    def _learn_conditionals(self, X_disc, ordering, parents,
                            n_samples, epsilon):
        """
        Learn noisy conditional distributions for each attribute
        given its parents.
        """
        eps_per_attr = epsilon / max(len(ordering), 1)
        conditionals = {}

        for col in ordering:
            parent_cols = parents[col]

            if len(parent_cols) == 0:
                # Unconditional: just a histogram
                hist = np.zeros(self.n_bins, dtype=np.float64)
                for i in range(n_samples):
                    b = min(int(X_disc[i, col]), self.n_bins - 1)
                    hist[b] += 1

                sensitivity = 1.0 / max(n_samples, 1)
                if eps_per_attr > 0:
                    sigma = sensitivity * np.sqrt(
                        2 * np.log(1.25 / 1e-5)
                    ) / eps_per_attr
                    hist += np.random.normal(0, sigma * n_samples, hist.shape)

                hist = np.maximum(hist, 0)
                total = hist.sum()
                if total > 0:
                    hist = hist / total
                else:
                    hist = np.ones(self.n_bins) / self.n_bins

                conditionals[col] = {"type": "marginal", "dist": hist}

            else:
                # Conditional on parent
                p_col = parent_cols[0]
                # Build conditional table: P(col | parent)
                cond_table = np.zeros((self.n_bins, self.n_bins),
                                      dtype=np.float64)
                for i in range(n_samples):
                    pc = min(int(X_disc[i, p_col]), self.n_bins - 1)
                    cc = min(int(X_disc[i, col]), self.n_bins - 1)
                    cond_table[pc, cc] += 1

                # Add noise
                sensitivity = 1.0 / max(n_samples, 1)
                if eps_per_attr > 0:
                    sigma = sensitivity * np.sqrt(
                        2 * np.log(1.25 / 1e-5)
                    ) / eps_per_attr
                    cond_table += np.random.normal(
                        0, sigma * n_samples, cond_table.shape
                    )

                cond_table = np.maximum(cond_table, 0)
                # Normalize each row (condition)
                for r in range(self.n_bins):
                    row_sum = cond_table[r].sum()
                    if row_sum > 0:
                        cond_table[r] /= row_sum
                    else:
                        cond_table[r] = np.ones(self.n_bins) / self.n_bins

                conditionals[col] = {
                    "type": "conditional",
                    "dist": cond_table,
                    "parent": p_col,
                }

        return conditionals

    def _sample_bn(self, ordering, conditionals, all_edges, n_samples, d):
        """Sample from the learned Bayesian network."""
        X_disc = np.zeros((n_samples, d), dtype=int)

        for col in ordering:
            cond = conditionals[col]

            if cond["type"] == "marginal":
                dist = cond["dist"]
                dist = np.maximum(dist, 0)
                total = dist.sum()
                if total > 0:
                    dist = dist / total
                else:
                    dist = np.ones(self.n_bins) / self.n_bins
                X_disc[:, col] = np.random.choice(
                    self.n_bins, size=n_samples, p=dist
                )
            else:
                p_col = cond["parent"]
                for i in range(n_samples):
                    pv = int(X_disc[i, p_col])
                    pv = min(pv, self.n_bins - 1)
                    dist = cond["dist"][pv]
                    dist = np.maximum(dist, 0)
                    total = dist.sum()
                    if total > 0:
                        dist = dist / total
                    else:
                        dist = np.ones(self.n_bins) / self.n_bins
                    X_disc[i, col] = np.random.choice(self.n_bins, p=dist)

        # Convert discrete bins back to continuous values
        X_cont = np.zeros((n_samples, d), dtype=np.float32)
        for j in range(d):
            edges = all_edges[j]
            bin_idx = X_disc[:, j]
            # Sample uniformly within each bin
            lo = edges[bin_idx]
            hi = edges[np.minimum(bin_idx + 1, len(edges) - 1)]
            X_cont[:, j] = np.random.uniform(lo, hi).astype(np.float32)

        return X_cont

    def fit(self, parties, epsilon=1.0, **kwargs):
        K = len(parties)
        d = parties[0]["X"].shape[1]
        print(f"[PrivBayes] Training with K={K} parties, eps={epsilon}, d={d}")

        self.party_models = []
        self.party_sizes = []

        for k in range(K):
            X_k = parties[k]["X"]
            n_k = X_k.shape[0]
            print(f"  Party {k}/{K}: {n_k} samples")

            # Discretize
            X_disc, all_edges = self._discretize(X_k)

            # Split epsilon: half for ordering, half for conditionals
            eps_order = epsilon / 2.0
            eps_cond = epsilon / 2.0

            # Learn ordering
            ordering, parents_map = self._learn_ordering(
                X_disc, n_k, eps_order
            )

            # Learn conditionals
            conditionals = self._learn_conditionals(
                X_disc, ordering, parents_map, n_k, eps_cond
            )

            self.party_models.append({
                "ordering": ordering,
                "conditionals": conditionals,
                "all_edges": all_edges,
                "d": d,
            })
            self.party_sizes.append(n_k)

        print(f"[PrivBayes] Training complete.")

    def generate(self, n_samples):
        if not self.party_models:
            raise RuntimeError("Must call .fit() before .generate()")

        total = sum(self.party_sizes)
        all_samples = []

        for k, pm in enumerate(self.party_models):
            n_k = int(round(n_samples * self.party_sizes[k] / total))
            if n_k == 0:
                n_k = 1
            samples = self._sample_bn(
                pm["ordering"], pm["conditionals"],
                pm["all_edges"], n_k, pm["d"]
            )
            all_samples.append(samples)

        result = np.concatenate(all_samples, axis=0)
        if result.shape[0] > n_samples:
            result = result[:n_samples]
        elif result.shape[0] < n_samples:
            extra = n_samples - result.shape[0]
            idx = np.random.choice(result.shape[0], extra, replace=True)
            result = np.concatenate([result, result[idx]], axis=0)

        return result


# ===================================================================
# 5. CTGAN Generator (wrapper, graceful skip if not installed)
# ===================================================================

class CTGANGenerator(BaseGenerator):
    """
    CTGAN wrapper. Uses the ctgan/sdv library if installed.
    Falls back gracefully with a warning if not available.
    """

    def __init__(self):
        super().__init__(name="CTGAN")
        self._available = False
        self._generator = None
        self.party_sizes = []
        self.party_models = []
        self.d = None

        try:
            from ctgan import CTGAN
            self._available = True
        except ImportError:
            warnings.warn(
                "[CTGAN] ctgan package not installed. "
                "Install with: pip install ctgan. "
                "CTGAN experiments will be skipped."
            )

    def fit(self, parties, epsilon=1.0, epochs=50, **kwargs):
        if not self._available:
            print("[CTGAN] Skipping - ctgan package not installed.")
            return

        from ctgan import CTGAN

        K = len(parties)
        self.d = parties[0]["X"].shape[1]
        print(f"[CTGAN] Training with K={K} parties, epochs={epochs}")

        self.party_models = []
        self.party_sizes = []

        for k in range(K):
            X_k = parties[k]["X"]
            n_k = X_k.shape[0]
            print(f"  Party {k}/{K}: {n_k} samples")

            # CTGAN expects DataFrame
            df = pd.DataFrame(X_k, columns=[f"f{i}" for i in range(self.d)])

            ctgan = CTGAN(epochs=epochs, verbose=False)
            ctgan.fit(df)

            self.party_models.append(ctgan)
            self.party_sizes.append(n_k)

        print(f"[CTGAN] Training complete.")

    def generate(self, n_samples):
        if not self._available or not self.party_models:
            raise RuntimeError(
                "CTGAN not available or not fitted. "
                "Install ctgan or call .fit() first."
            )

        total = sum(self.party_sizes)
        all_samples = []

        for k, ctgan in enumerate(self.party_models):
            n_k = int(round(n_samples * self.party_sizes[k] / total))
            if n_k == 0:
                n_k = 1
            df_syn = ctgan.sample(n_k)
            all_samples.append(df_syn.values.astype(np.float32))

        result = np.concatenate(all_samples, axis=0)
        if result.shape[0] > n_samples:
            result = result[:n_samples]
        elif result.shape[0] < n_samples:
            extra = n_samples - result.shape[0]
            idx = np.random.choice(result.shape[0], extra, replace=True)
            result = np.concatenate([result, result[idx]], axis=0)

        return result


# ===================================================================
# Generator registry
# ===================================================================

def get_generator(name, **kwargs):
    """
    Factory function to create a generator by name.

    Parameters
    ----------
    name : str
        One of: 'fedsynth', 'fedavg', 'fedavg_8bit', 'independent',
        'centralized', 'privbayes', 'ctgan'

    Returns
    -------
    BaseGenerator instance
    """
    name_lower = name.lower().replace(" ", "").replace("-", "").replace("_", "")

    if name_lower == "fedsynth":
        return CrossSynthGenerator(**kwargs)
    elif name_lower == "fedavg":
        return FedAvgGenerator(compression="none", **kwargs)
    elif name_lower in ("fedavg8bit", "8bitfedavg"):
        return FedAvgGenerator(compression="int8", **kwargs)
    elif name_lower == "independent":
        return IndependentGenerator(**kwargs)
    elif name_lower == "centralized":
        return CentralizedGenerator(**kwargs)
    elif name_lower == "privbayes":
        return PrivBayesGenerator(**kwargs)
    elif name_lower == "ctgan":
        return CTGANGenerator()
    else:
        raise ValueError(
            f"Unknown generator '{name}'. "
            "Choose from: fedsynth, fedavg, fedavg_8bit, independent, "
            "centralized, privbayes, ctgan"
        )


# ===================================================================
# Quick smoke test
# ===================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # Create dummy data (3 parties)
    d = 8
    parties = [
        {"X": np.random.randn(200, d).astype(np.float32), "y": np.zeros(200)},
        {"X": np.random.randn(200, d).astype(np.float32), "y": np.zeros(200)},
        {"X": np.random.randn(200, d).astype(np.float32), "y": np.zeros(200)},
    ]

    # Test CrossSynth
    print("=" * 60)
    print("Testing CrossSynth")
    print("=" * 60)
    gen = CrossSynthGenerator(hidden_dim=64, n_layers=2, n_timesteps=50,
                            cond_dim=64)
    gen.fit(parties, epsilon=1.0, epochs=3, verbose=False)
    syn = gen.generate(100)
    print(f"Generated shape: {syn.shape}")
    print(f"Mean: {syn.mean(axis=0)[:3]}")

    # Test Independent
    print("\n" + "=" * 60)
    print("Testing Independent")
    print("=" * 60)
    gen2 = IndependentGenerator(hidden_dim=64, n_layers=2, n_timesteps=50)
    gen2.fit(parties, epsilon=1.0, epochs=3, verbose=False)
    syn2 = gen2.generate(100)
    print(f"Generated shape: {syn2.shape}")

    # Test Centralized
    print("\n" + "=" * 60)
    print("Testing Centralized")
    print("=" * 60)
    gen3 = CentralizedGenerator(hidden_dim=64, n_layers=2, n_timesteps=50)
    gen3.fit(parties, epsilon=1.0, epochs=3, verbose=False)
    syn3 = gen3.generate(100)
    print(f"Generated shape: {syn3.shape}")

    # Test PrivBayes
    print("\n" + "=" * 60)
    print("Testing PrivBayes")
    print("=" * 60)
    gen4 = PrivBayesGenerator(n_bins=10)
    gen4.fit(parties, epsilon=1.0)
    syn4 = gen4.generate(100)
    print(f"Generated shape: {syn4.shape}")
    print(f"Mean: {syn4.mean(axis=0)[:3]}")

    # Test CTGAN
    print("\n" + "=" * 60)
    print("Testing CTGAN")
    print("=" * 60)
    gen5 = CTGANGenerator()
    if gen5._available:
        gen5.fit(parties, epsilon=1.0, epochs=3)
        syn5 = gen5.generate(100)
        print(f"Generated shape: {syn5.shape}")
    else:
        print("CTGAN not available, skipping.")

    print("\nAll generator tests passed!")
