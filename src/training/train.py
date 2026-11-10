"""Training loop with KL warm-up, gradient clipping, label-free early stopping and best-checkpoint selection."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.vae import VAE, VAEConfig
from src.training.losses import beta_at_epoch, kl_divergence, reconstruction_sse, vae_loss


@dataclass
class TrainConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 100
    early_stopping_patience: int = 10
    kl_warmup_epochs: int = 10
    beta: float = 1.0
    grad_clip_norm: float = 5.0
    val_noise_seed: int = 1234
    variational: bool = True   # False = plain autoencoder: z = mu, no sampling, no KL term (ablation)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**d)


def make_loader(X: np.ndarray, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    """Single-process DataLoader whose shuffling order is fixed by its own seeded generator."""
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate_loss(model: VAE, X: np.ndarray, device: torch.device, beta: float, noise_seed: int,
                  batch_size: int = 8192, variational: bool = True) -> dict[str, float]:
    """Validation loss with fixed sampling noise, plus the deterministic (z = mu) reconstruction MSE."""
    model.eval()
    generator = torch.Generator().manual_seed(noise_seed)
    sums = {"recon": 0.0, "kl": 0.0, "recon_mse": 0.0, "recon_mse_det": 0.0}
    n = len(X)
    for start in range(0, n, batch_size):
        x = torch.from_numpy(X[start:start + batch_size]).to(device)
        out = model(x, sample=variational, generator=generator)
        sums["recon"] += reconstruction_sse(x, out.recon).sum().item()
        sums["kl"] += kl_divergence(out.mu, out.logvar).sum().item()
        sums["recon_mse"] += (x - out.recon).pow(2).mean(dim=1).sum().item()
        det = model.decode(out.mu)
        sums["recon_mse_det"] += (x - det).pow(2).mean(dim=1).sum().item()
    stats = {k: v / n for k, v in sums.items()}
    stats["total"] = stats["recon"] + beta * stats["kl"]
    return stats


@torch.no_grad()
def kl_per_dimension(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    """Average KL contribution of each latent dimension (a dimension with KL ~ 0 is unused)."""
    model.eval()
    total = torch.zeros(model.cfg.latent_dim, dtype=torch.float64)
    for start in range(0, len(X), batch_size):
        mu, logvar = model.encode(torch.from_numpy(X[start:start + batch_size]).to(device))
        total += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=0).double().cpu()
    return (total / len(X)).numpy()


def train_vae(model_cfg: VAEConfig, train_cfg: TrainConfig, X_train: np.ndarray, X_val: np.ndarray, seed: int,
              device: torch.device, log: Callable[[str], None] = print) -> dict:
    """Train one VAE. Early stopping monitors the validation negative ELBO (beta = final beta) after warm-up."""
    torch.manual_seed(seed)
    model = VAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    loader = make_loader(X_train, train_cfg.batch_size, seed)

    history = []
    best_state, best_epoch, best_val = None, 0, float("inf")
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    for epoch in range(1, train_cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        beta = beta_at_epoch(epoch, train_cfg.kl_warmup_epochs, train_cfg.beta) if train_cfg.variational else 0.0
        model.train()
        sums = {"total": 0.0, "recon": 0.0, "kl": 0.0, "recon_mse": 0.0}
        grad_norms = []
        for (x,) in loader:
            x = x.to(device)
            losses = vae_loss(x, model(x, sample=train_cfg.variational), beta)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm).item())
            optimizer.step()
            for k in sums:
                sums[k] += losses[k].item() * len(x)
        train_stats = {k: v / len(X_train) for k, v in sums.items()}
        val_stats = evaluate_loss(model, X_val, device, train_cfg.beta if train_cfg.variational else 0.0,
                                  train_cfg.val_noise_seed, variational=train_cfg.variational)
        warm = epoch >= train_cfg.kl_warmup_epochs

        improved = warm and val_stats["total"] < best_val - 1e-6
        if improved:
            best_val, best_epoch = val_stats["total"], epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        elif warm:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch, "beta": beta,
            "train_total": train_stats["total"], "train_recon": train_stats["recon"], "train_kl": train_stats["kl"],
            "train_recon_mse": train_stats["recon_mse"],
            "val_total": val_stats["total"], "val_recon": val_stats["recon"], "val_kl": val_stats["kl"],
            "val_recon_mse": val_stats["recon_mse"], "val_recon_mse_det": val_stats["recon_mse_det"],
            "grad_norm_mean": float(np.mean(grad_norms)), "grad_clipped_share": float(np.mean(np.array(grad_norms) > train_cfg.grad_clip_norm)),
            "is_best": improved, "seconds": time.perf_counter() - epoch_start,
        })
        if epoch == 1 or epoch % 10 == 0 or improved:
            log(f"epoch {epoch:3d} beta={beta:.2f} train={train_stats['total']:.4f} (recon {train_stats['recon']:.4f}, kl {train_stats['kl']:.4f}) "
                f"val={val_stats['total']:.4f}{' *' if improved else ''}")
        if warm and epochs_without_improvement >= train_cfg.early_stopping_patience:
            log(f"early stopping at epoch {epoch}; best epoch {best_epoch} (val {best_val:.4f})")
            break

    if best_state is None:  # max_epochs < warm-up length; fall back to the last epoch
        best_state, best_epoch = copy.deepcopy(model.state_dict()), history[-1]["epoch"]
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "epochs_run": history[-1]["epoch"],
        "runtime_seconds": time.perf_counter() - start_time,
    }
