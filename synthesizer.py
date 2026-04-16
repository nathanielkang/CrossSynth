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
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from marginals import (
    compute_1way_marginals_shared,
    compute_2way_marginals,
    compute_shared_bin_edges,
    add_dp_noise_1way,
    aggregate_marginals_1way,
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

    def train_model(self, X_train, cond_vec=None, epochs=50,
                    batch_size=256, lr=1e-3, verbose=True):
        """
        Train the diffusion model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, d)
        cond_vec : np.ndarray, shape (cond_dim,) or None
            If provided, this conditioning vector is broadcast to all samples.
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

        # Prepare conditioning: broadcast to all samples
        cond_tensor = None
        if cond_vec is not None and self.cond_dim > 0:
            cond_tensor = torch.tensor(cond_vec, dtype=torch.float32)
            cond_tensor = cond_tensor.unsqueeze(0).expand(X_tensor.shape[0], -1)

        if cond_tensor is not None:
            dataset = TensorDataset(X_tensor, cond_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
                loss = self.compute_loss(x_batch, c_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        return losses

    @torch.no_grad()
    def sample(self, n_samples, cond_vec=None, verbose=False):
        """
        Generate synthetic rows via DDPM reverse process.

        Parameters
        ----------
        n_samples : int
        cond_vec : np.ndarray, shape (cond_dim,) or None
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
            cond = cond.unsqueeze(0).expand(n_samples, -1)

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


# ===================================================================
# 1. CrossSynth Generator (our method)
# ===================================================================

class CrossSynthGenerator(BaseGenerator):
    """
    CrossSynth: each party trains a local diffusion model conditioned on
    securely aggregated cross-party marginals.

    Steps:
        1. All parties agree on shared bin edges.
        2. Each party computes noisy 1-way marginals (Gaussian mechanism).
        3. Noisy marginals are aggregated (simulated secure aggregation).
        4. Aggregated marginals are encoded into a fixed-size vector.
        5. Each party trains a local TabDDPM conditioned on this vector.
        6. Generation: sample from each party's model using the aggregated
           conditioning vector, then union all samples.
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

    def fit(self, parties, epsilon=1.0, epochs=50, batch_size=256,
            lr=1e-3, verbose=True, delta=1e-5, **kwargs):
        """
        Train CrossSynth across all parties.

        Privacy budget allocation:
            - epsilon/2 for marginal computation
            - epsilon/2 for local training (no additional DP on gradients
              since the marginals are the only shared information)
        """
        K = len(parties)
        d = parties[0]["X"].shape[1]
        eps_marginals = epsilon / 2.0

        print(f"[CrossSynth] Training with K={K} parties, eps={epsilon}, "
              f"d={d}, epochs={epochs}")

        # Step 1: Agree on shared bin edges
        shared_edges = compute_shared_bin_edges(parties, n_bins=self.n_bins)

        # Step 2: Each party computes noisy 1-way marginals
        all_noisy_marginals = []
        for k in range(K):
            m1 = compute_1way_marginals_shared(parties[k]["X"], shared_edges)
            m1_noisy = add_dp_noise_1way(
                m1, n_samples=parties[k]["X"].shape[0],
                epsilon=eps_marginals, delta=delta
            )
            all_noisy_marginals.append(m1_noisy)

        # Step 3: Aggregate marginals
        agg_marginals = aggregate_marginals_1way(all_noisy_marginals)

        # Step 4: Encode into conditioning vector
        self.cond_vec = encode_marginals(agg_marginals, max_dim=self.cond_dim)

        # Normalize ALL features for diffusion training (zero mean, unit std).
        # Categorical features are label-encoded integers and need scaling too.
        X_all_parts = np.concatenate([p["X"] for p in parties], axis=0)
        self.data_mean = X_all_parts.mean(axis=0).astype(np.float32)
        self.data_std = X_all_parts.std(axis=0).astype(np.float32)
        self.data_std[self.data_std < 1e-8] = 1.0

        # Step 5: Each party trains a local conditioned diffusion model
        self.models = []
        self.party_sizes = []

        for k in range(K):
            print(f"  Party {k}/{K}: {parties[k]['X'].shape[0]} samples")
            X_norm = (parties[k]["X"] - self.data_mean) / self.data_std
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
            )
            self.models.append(model)
            self.party_sizes.append(parties[k]["X"].shape[0])

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
            samples = model.sample(n_k, cond_vec=self.cond_vec, verbose=False)
            samples = samples * self.data_std + self.data_mean
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
# 2. Independent Generator (no cross-party info)
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

    def fit(self, parties, epsilon=1.0, epochs=50, batch_size=256,
            lr=1e-3, verbose=True, **kwargs):
        K = len(parties)
        d = parties[0]["X"].shape[1]
        print(f"[Independent] Training with K={K} parties, eps={epsilon}, "
              f"d={d}, epochs={epochs}")

        # Normalize ALL features for diffusion training (zero mean, unit std)
        X_all_parts = np.concatenate([p["X"] for p in parties], axis=0)
        self.data_mean = X_all_parts.mean(axis=0).astype(np.float32)
        self.data_std = X_all_parts.std(axis=0).astype(np.float32)
        self.data_std[self.data_std < 1e-8] = 1.0

        self.models = []
        self.party_sizes = []

        for k in range(K):
            print(f"  Party {k}/{K}: {parties[k]['X'].shape[0]} samples")
            X_norm = (parties[k]["X"] - self.data_mean) / self.data_std
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
            samples = samples * self.data_std + self.data_mean
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
        One of: 'fedsynth', 'independent', 'centralized', 'privbayes', 'ctgan'

    Returns
    -------
    BaseGenerator instance
    """
    name_lower = name.lower().replace(" ", "").replace("-", "").replace("_", "")

    if name_lower == "fedsynth":
        return CrossSynthGenerator(**kwargs)
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
            f"Choose from: fedsynth, independent, centralized, privbayes, ctgan"
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
