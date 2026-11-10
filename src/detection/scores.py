"""Anomaly scores computed from a trained VAE (higher = more unusual).

S1  recon_mse : mean over features of (x - decoder(mu(x)))^2. The primary score: a reconstruction
                distance, NOT a probability.
S2  neg_elbo  : 1/2 * ||x - x_hat||^2 (averaged over L samples of z) + KL + const. An upper bound on
                -log p(x) under the model.
S3  iwae_nll  : -log (1/K) sum_k p(x|z_k) p(z_k) / q(z_k|x). A tighter estimate of -log p(x).
None of these is P(fraud | x).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.models.vae import VAE
from src.training.losses import kl_divergence

LOG_2PI = math.log(2 * math.pi)


def _batches(X: np.ndarray, batch_size: int):
    for start in range(0, len(X), batch_size):
        yield torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32))


@torch.no_grad()
def recon_mse(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for x in _batches(X, batch_size):
        x = x.to(device)
        recon = model(x, sample=False).recon
        out.append((x - recon).pow(2).mean(dim=1).double().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def per_feature_squared_error(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for x in _batches(X, batch_size):
        x = x.to(device)
        out.append((x - model(x, sample=False).recon).pow(2).double().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def neg_elbo(model: VAE, X: np.ndarray, device: torch.device, n_samples: int = 10, seed: int = 0,
             batch_size: int = 4096) -> np.ndarray:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    dim = model.cfg.input_dim
    out = []
    for x in _batches(X, batch_size):
        x = x.to(device)
        mu, logvar = model.encode(x)
        recon_term = torch.zeros(len(x), device=device)
        for _ in range(n_samples):
            z = model.reparameterize(mu, logvar, generator)
            recon_term += 0.5 * (x - model.decode(z)).pow(2).sum(dim=1)
        score = recon_term / n_samples + kl_divergence(mu, logvar) + 0.5 * dim * LOG_2PI
        out.append(score.double().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def iwae_nll(model: VAE, X: np.ndarray, device: torch.device, n_samples: int = 100, seed: int = 0,
             batch_size: int = 1024) -> np.ndarray:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    d, D = model.cfg.latent_dim, model.cfg.input_dim
    out = []
    for x in _batches(X, batch_size):
        x = x.to(device)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((n_samples, *mu.shape), generator=generator).to(device)
        z = mu.unsqueeze(0) + std.unsqueeze(0) * eps                       # (K, B, d)
        recon = model.decode(z.reshape(-1, d)).reshape(n_samples, len(x), D)
        log_px_z = -0.5 * (x.unsqueeze(0) - recon).pow(2).sum(-1) - 0.5 * D * LOG_2PI
        log_pz = -0.5 * z.pow(2).sum(-1) - 0.5 * d * LOG_2PI
        log_qz = -0.5 * (eps.pow(2) + logvar.unsqueeze(0)).sum(-1) - 0.5 * d * LOG_2PI
        log_w = log_px_z + log_pz - log_qz
        nll = -(torch.logsumexp(log_w, dim=0) - math.log(n_samples))
        out.append(nll.double().cpu())
    return torch.cat(out).numpy()


def all_scores(model: VAE, X: np.ndarray, device: torch.device, score_cfg: dict) -> dict[str, np.ndarray]:
    return {
        "recon_mse": recon_mse(model, X, device),
        "neg_elbo": neg_elbo(model, X, device, score_cfg["elbo_samples"], score_cfg["seed"]),
        "iwae_nll": iwae_nll(model, X, device, score_cfg["iwae_samples"], score_cfg["seed"]),
    }
