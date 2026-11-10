"""Comparative and post-hoc analysis of trained models (run only after test predictions are saved)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.vae import VAE


@torch.no_grad()
def latent_means(model: VAE, X: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Posterior means mu and log-variances for every row (CPU)."""
    model.eval()
    mus, lvs = [], []
    for start in range(0, len(X), batch_size):
        mu, lv = model.encode(torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)))
        mus.append(mu.numpy())
        lvs.append(lv.numpy())
    return np.concatenate(mus), np.concatenate(lvs)


@torch.no_grad()
def reconstructions(model: VAE, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        out.append(model(torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)), sample=False).recon.numpy())
    return np.concatenate(out)


def score_distribution_table(scores: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """Quantiles of the anomaly score per class, plus the fraud/normal median ratio (separation)."""
    rows = []
    for name, s in scores.items():
        n, f = s[y == 0], s[y == 1]
        rows.append({"model": name,
                     "normal_median": np.median(n), "normal_p99": np.percentile(n, 99),
                     "fraud_median": np.median(f), "fraud_p25": np.percentile(f, 25), "fraud_p75": np.percentile(f, 75),
                     "median_ratio_fraud_over_normal": np.median(f) / np.median(n),
                     "share_of_frauds_above_normal_p99": float(np.mean(f > np.percentile(n, 99)))})
    return pd.DataFrame(rows).set_index("model")


def per_feature_error_table(X: np.ndarray, recon: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    sq = (X - recon) ** 2
    t = pd.DataFrame({"normal_mean_sq_error": sq[y == 0].mean(0), "fraud_mean_sq_error": sq[y == 1].mean(0)}, index=names)
    t["fraud_share_of_total_error"] = t["fraud_mean_sq_error"] / t["fraud_mean_sq_error"].sum()
    t["ratio_fraud_over_normal"] = t["fraud_mean_sq_error"] / t["normal_mean_sq_error"]
    return t.sort_values("fraud_mean_sq_error", ascending=False)


def latent_summary(mu: np.ndarray, y: np.ndarray) -> dict:
    norm = np.linalg.norm(mu, axis=1)
    centroid_n, centroid_f = mu[y == 0].mean(0), mu[y == 1].mean(0)
    return {"median_norm_normal": float(np.median(norm[y == 0])), "median_norm_fraud": float(np.median(norm[y == 1])),
            "centroid_distance": float(np.linalg.norm(centroid_f - centroid_n)),
            "mean_within_normal_spread": float(np.linalg.norm(mu[y == 0] - centroid_n, axis=1).mean())}


def detection_overlap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    fa, fb = (pred_a == 1) & (y == 1), (pred_b == 1) & (y == 1)
    return {"frauds": int(y.sum()), "caught_by_a": int(fa.sum()), "caught_by_b": int(fb.sum()), "caught_by_both": int((fa & fb).sum()),
            "only_a": int((fa & ~fb).sum()), "only_b": int((fb & ~fa).sum()), "missed_by_both": int(((~fa) & (~fb) & (y == 1)).sum())}


def precision_vs_prevalence(tpr: float, fpr: float, prevalences: np.ndarray) -> np.ndarray:
    """Expected precision of a detector with fixed TPR/FPR as the fraud rate changes (Bayes' rule)."""
    return tpr * prevalences / (tpr * prevalences + fpr * (1 - prevalences))
