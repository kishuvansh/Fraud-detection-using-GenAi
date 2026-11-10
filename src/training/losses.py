"""VAE objective (negative ELBO) for a Gaussian decoder with unit variance.

Per sample:
    reconstruction = 1/2 * sum_j (x_j - x_hat_j)^2            (= -log p(x|z) + const)
    KL             = -1/2 * sum_k (1 + logvar_k - mu_k^2 - exp(logvar_k))
    loss           = reconstruction + beta * KL

The reconstruction term is a squared error (an MSE summed over the 29 features and halved), which is
exactly the Gaussian log-likelihood with sigma = 1. Summing over features, rather than averaging,
keeps reconstruction and KL on the scale the ELBO defines. Averaging would shrink reconstruction by
29/2 relative to KL and push the model towards posterior collapse. The per-feature MSE is also logged
because it is easier to read.
"""

from __future__ import annotations

import torch

from src.models.vae import VAEOutput


def reconstruction_sse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Half the squared error summed over features, per sample (shape (n,))."""
    return 0.5 * (x - recon).pow(2).sum(dim=1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, sigma^2) || N(0, I)) per sample (shape (n,))."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def vae_loss(x: torch.Tensor, out: VAEOutput, beta: float) -> dict[str, torch.Tensor]:
    recon = reconstruction_sse(x, out.recon)
    kl = kl_divergence(out.mu, out.logvar)
    total = (recon + beta * kl).mean()
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite VAE loss")
    return {
        "total": total,
        "recon": recon.mean(),
        "kl": kl.mean(),
        "recon_mse": (x - out.recon).pow(2).mean(),
    }


def beta_at_epoch(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear KL warm-up: beta = beta_max * epoch / warmup for epochs 1..warmup, then beta_max."""
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, epoch / warmup_epochs)
