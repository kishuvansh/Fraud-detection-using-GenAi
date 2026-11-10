"""Figures for training, threshold calibration, detection and generation (shared style with EDA plots).

Colour roles: normal = blue, fraud = orange, synthetic = aqua; train = blue, validation = aqua.
No chart uses two y-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.plots import DIVERGING, FRAUD_COLOR, NORMAL_COLOR, SURFACE, TEXT, TEXT_SECONDARY, _save

SYNTH_COLOR = "#1baf7a"
TRAIN_COLOR = NORMAL_COLOR
VAL_COLOR = SYNTH_COLOR
NEUTRAL = "#6f6e69"
GRID_SHADE = "#efeee9"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])


# ----------------------------------------------------------------------------- training
def plot_training_curves(history: pd.DataFrame, best_epoch: int, warmup: int, title: str, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    panels = [("total", "Total loss (negative ELBO)\ntrain uses current β, validation uses β = 1"),
              ("recon", "Reconstruction term ½‖x − x̂‖²"), ("kl", "KL divergence")]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(history["epoch"], history[f"train_{key}"], color=TRAIN_COLOR, linewidth=2, label="train")
        ax.plot(history["epoch"], history[f"val_{key}"], color=VAL_COLOR, linewidth=2, label="validation")
        ax.axvspan(0.5, warmup + 0.5, color=GRID_SHADE, zorder=0)
        ax.axvline(best_epoch, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
    axes[0].annotate(f"best epoch {best_epoch}", (best_epoch, axes[0].get_ylim()[1]), xytext=(-4, -12), textcoords="offset points",
                     ha="right", fontsize=8, color=TEXT_SECONDARY)
    axes[0].annotate("KL warm-up", (1, axes[0].get_ylim()[1]), xytext=(2, -12), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    if history["train_total"].iloc[0] > 3 * history["train_total"].iloc[-1]:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    axes[2].legend(loc="lower right")
    fig.suptitle(title)
    return _save(fig, path)



def plot_seed_curves(histories: dict[int, pd.DataFrame], best_epochs: dict[int, int], title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for seed, h in histories.items():
        ax.plot(h["epoch"], h["val_total"], color=VAL_COLOR, linewidth=1.4, alpha=0.75)
        ax.scatter([best_epochs[seed]], [h.loc[h["epoch"] == best_epochs[seed], "val_total"].iloc[0]], color=TEXT, s=12, zorder=3)
    ax.set_ylim(top=np.percentile(np.concatenate([h["val_total"].to_numpy() for h in histories.values()]), 97))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation negative ELBO (β = 1)")
    ax.set_title(f"{len(histories)} seeds; dots = selected (best) epoch", fontsize=10)
    fig.suptitle(title)
    return _save(fig, path)


def plot_validation_scores(val_scores: np.ndarray, tau: float, percentile: float, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    bins = np.logspace(np.log10(max(val_scores.min(), 1e-4)), np.log10(val_scores.max()), 80)
    ax.hist(val_scores, bins=bins, color=NORMAL_COLOR, edgecolor=SURFACE, linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(tau, color=TEXT, linewidth=1.5, linestyle="--")
    ax.annotate(f"threshold τ = {tau:.3f}\n({percentile:g}th percentile:\n{100 - percentile:g}% of these rows above)", (tau, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", va="top", fontsize=8, color=TEXT)
    for q in (50, 90):
        v = np.percentile(val_scores, q)
        ax.axvline(v, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        ax.annotate(f"p{q}", (v, ax.get_ylim()[1] * 0.6), xytext=(3, 0), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Reconstruction error (per-feature MSE, log scale)")
    ax.set_ylabel("Validation transactions")
    fig.suptitle(title)
    return _save(fig, path)


# ----------------------------------------------------------------------------- detection
def plot_pr_curve(curves: dict[str, tuple[np.ndarray, np.ndarray, str]], prevalence: float, operating_points: dict[str, tuple[float, float]],
                  title: str, path: Path | None = None) -> plt.Figure:
    """curves: label -> (y, scores, colour). operating_points: label -> (recall, precision)."""
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for label, (y, s, color) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=color, linewidth=2, label=label)
    for label, (rec, prec) in operating_points.items():
        ax.scatter([rec], [prec], s=70, color=TEXT, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.annotate(label, (rec, prec), xytext=(8, 6), textcoords="offset points", fontsize=8, color=TEXT)
    ax.axhline(prevalence, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.annotate(f"random scorer (precision = prevalence {100 * prevalence:.2f}%)", (0.02, prevalence), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.legend(loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_roc_curve(curves: dict[str, tuple[np.ndarray, np.ndarray, str]], operating_points: dict[str, tuple[float, float]],
                   title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    for label, (y, s, color) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    for label, (fpr_v, tpr_v) in operating_points.items():
        ax.scatter([fpr_v], [tpr_v], s=70, color=TEXT, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.annotate(label, (fpr_v, tpr_v), xytext=(8, -12), textcoords="offset points", fontsize=8, color=TEXT)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.legend(loc="lower right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_confusion_matrix(m: dict, title: str, path: Path | None = None) -> plt.Figure:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    row_share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(row_share, cmap=SEQ_BLUE, vmin=0, vmax=1)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({100 * row_share[i, j]:.1f}% of row)", ha="center", va="center",
                    fontsize=10, color=SURFACE if row_share[i, j] > 0.5 else TEXT)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual fraud"])
    ax.grid(visible=False)
    ax.set_title(f"Threshold τ = {m['threshold']:.3f}  ·  precision {m['precision']:.3f}  ·  recall {m['recall']:.3f}", fontsize=9)
    fig.suptitle(title)
    return _save(fig, path)


def plot_score_by_class(y: np.ndarray, scores: np.ndarray, tau: float, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    bins = np.logspace(np.log10(max(scores.min(), 1e-4)), np.log10(scores.max()), 70)
    for k, color, label in ((0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")):
        ax.hist(scores[y == k], bins=bins, density=True, histtype="step", linewidth=2, color=color, label=f"{label} (n = {int((y == k).sum()):,})")
    ax.set_xscale("log")
    ax.axvline(tau, color=TEXT, linewidth=1.5, linestyle="--")
    ax.annotate(f"τ = {tau:.3f}", (tau, ax.get_ylim()[1] * 0.92), xytext=(5, 0), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlabel("Reconstruction error (per-feature MSE, log scale)")
    ax.set_ylabel("Density within class")
    ax.legend(loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_score_distribution(test_scores: np.ndarray, val_scores: np.ndarray, tau: float, title: str, path: Path | None = None) -> plt.Figure:
    """Distribution of the anomaly score itself (no labels): all test rows vs normal validation rows."""
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    for s, color, label in ((val_scores, VAL_COLOR, "Validation (calibration rows)"), (test_scores, NEUTRAL, "Test (all rows, unlabelled)")):
        v = np.sort(s)
        ax.plot(v, 1 - np.arange(1, len(v) + 1) / len(v), color=color, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(tau, color=TEXT, linewidth=1.5, linestyle="--")
    flagged = np.mean(test_scores > tau)
    ax.annotate(f"τ = {tau:.3f}: flags {100 * flagged:.2f}% of test rows", (tau, 0.3), xytext=(6, 0), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlabel("Anomaly score (reconstruction error, log scale)")
    ax.set_ylabel("Share of rows with a higher score (log)")
    ax.legend(loc="lower left")
    fig.suptitle(title)
    return _save(fig, path)


def plot_threshold_curves(y: np.ndarray, scores: np.ndarray, sensitivity: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    """Precision / recall / F1 across thresholds. Points = validation-percentile thresholds; star = primary."""
    p, r, t = precision_recall_curve(y, scores)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, layout="constrained")
    for ax, values, key, label in ((axes[0], p[:-1], "precision", "Precision"), (axes[1], r[:-1], "recall", "Recall"), (axes[2], f1[:-1], "f1", "F1-score")):
        ax.plot(t, values, color=NEUTRAL, linewidth=1.5)
        sens = sensitivity[sensitivity["role"] == "sensitivity"]
        prim = sensitivity[sensitivity["role"] == "primary"]
        ax.scatter(sens["threshold"], sens[key], color=NORMAL_COLOR, s=30, zorder=3, label="validation-percentile thresholds (sensitivity)")
        ax.scatter(prim["threshold"], prim[key], color=TEXT, marker="*", s=180, zorder=4, label="primary τ (99th percentile)")
        for _, row in sensitivity.iterrows():
            ax.annotate(f"p{row['validation_percentile']:g}", (row["threshold"], row[key]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel("Threshold on reconstruction error (log)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(title)
    return _save(fig, path)


def plot_seed_comparison(table: pd.DataFrame, metric: str, title: str, path: Path | None = None) -> plt.Figure:
    """table: rows = seeds, columns = experiments. Dots per seed + mean bar."""
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    colors = {"E1 (normal-only)": NORMAL_COLOR, "E2 (unsupervised)": NEUTRAL}
    for i, col in enumerate(table.columns):
        ax.bar(i, table[col].mean(), width=0.55, color=colors.get(col, NEUTRAL), alpha=0.35)
        ax.scatter(np.full(len(table), i), table[col], color=colors.get(col, NEUTRAL), s=30, zorder=3)
        ax.annotate(f"{table[col].mean():.3f} ± {table[col].std():.3f}", (i, table[col].max()), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1e-3, table.to_numpy().max()) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.suptitle(title)
    return _save(fig, path)


# ----------------------------------------------------------------------------- generation
def plot_real_vs_synthetic_grid(real: pd.DataFrame, synth: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    cols = list(real.columns)
    fig, axes = plt.subplots(6, 5, figsize=(16, 17), layout="constrained")
    for ax, col in zip(axes.flat, cols):
        lo, hi = np.percentile(real[col], [0.5, 99.5])
        if col == "Amount":
            r, s = np.log1p(real[col]), np.log1p(synth[col])
            lo, hi = np.percentile(r, [0.5, 99.5])
            name = "log1p(Amount)"
        else:
            r, s, name = real[col], synth[col], col
        bins = np.linspace(lo, hi, 60)
        ax.hist(np.clip(r, lo, hi), bins=bins, density=True, histtype="step", linewidth=1.6, color=NORMAL_COLOR, label="Real normal (test)")
        ax.hist(np.clip(s, lo, hi), bins=bins, density=True, histtype="step", linewidth=1.6, color=SYNTH_COLOR, label="Synthetic")
        ax.set_title(name)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.grid(visible=False)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=10)
    fig.suptitle(title)
    return _save(fig, path)


def plot_moment_comparison(moments: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    m = moments.drop(index="Amount", errors="ignore")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, stat in zip(axes, ("mean", "std")):
        x, y = m[f"real_{stat}"], m[f"synthetic_{stat}"]
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = 0.08 * (lim[1] - lim[0])
        ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax.scatter(x, y, color=SYNTH_COLOR, s=30, zorder=3)
        for name in (y - x).abs().sort_values(ascending=False).index[:5]:  # label only the five largest gaps
            ax.annotate(name, (x[name], y[name]), xytext=(4, 2), textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xlabel(f"Real normal {stat}")
        ax.set_ylabel(f"Synthetic {stat}")
        ax.set_title(f"Per-feature {stat} (V1–V28; dotted line = perfect match)", fontsize=10)
    fig.suptitle(title)
    return _save(fig, path)


def plot_distance_bars(distances: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), layout="constrained")
    for ax, col, label in ((axes[0], "ks_statistic", "Kolmogorov–Smirnov statistic (0 = identical)"),
                           (axes[1], "wasserstein_std_units", "Wasserstein-1 distance (standardised units)")):
        ax.bar(distances.index, distances[col], color=SYNTH_COLOR, width=0.7)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title)
    return _save(fig, path)


def plot_correlation_comparison(corr: dict, names: list[str], title: str, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), layout="constrained")
    mats = [("Real normal (test)", corr["real_matrix"]), ("Synthetic", corr["synthetic_matrix"]),
            ("Difference (synthetic − real)", corr["synthetic_matrix"] - corr["real_matrix"])]
    image = None
    for ax, (label, mat) in zip(axes, mats):
        image = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(names)), names, fontsize=6.5)
        ax.grid(visible=False)
        ax.set_title(label)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.suptitle(title)
    return _save(fig, path)


def plot_pca_overlay(proj: dict, title: str, path: Path | None = None, n: int = 5000, seed: int = 0) -> plt.Figure:
    rng = np.random.default_rng(seed)
    r = proj["real"][rng.choice(len(proj["real"]), min(n, len(proj["real"])), replace=False)]
    s = proj["synthetic"][rng.choice(len(proj["synthetic"]), min(n, len(proj["synthetic"])), replace=False)]
    lim_x = np.percentile(np.r_[r[:, 0], s[:, 0]], [0.5, 99.5])
    lim_y = np.percentile(np.r_[r[:, 1], s[:, 1]], [0.5, 99.5])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True, layout="constrained")
    for ax, data, color, label in ((axes[0], r, NORMAL_COLOR, "Real normal (test)"), (axes[1], s, SYNTH_COLOR, "Synthetic")):
        ax.scatter(r[:, 0], r[:, 1], s=3, color="#d9d8d3", alpha=0.6, linewidths=0, label="real (reference)" if data is s else None)
        ax.scatter(data[:, 0], data[:, 1], s=3, color=color, alpha=0.45, linewidths=0, label=label)
        ax.set_title(label)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xlabel(f"PC1 ({100 * proj['explained_variance_ratio'][0]:.1f}% of real variance)")
        ax.legend(loc="upper right", markerscale=4)
    axes[0].set_ylabel(f"PC2 ({100 * proj['explained_variance_ratio'][1]:.1f}%)")
    fig.suptitle(title)
    return _save(fig, path)


def plot_memorisation(dists: dict, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    hi = np.percentile(np.r_[dists["synthetic"], dists["holdout"]], 99.5)
    bins = np.linspace(0, hi, 70)
    ax.hist(np.clip(dists["holdout"], 0, hi), bins=bins, density=True, histtype="step", linewidth=2, color=NORMAL_COLOR, label="Real held-out → nearest training row")
    ax.hist(np.clip(dists["synthetic"], 0, hi), bins=bins, density=True, histtype="step", linewidth=2, color=SYNTH_COLOR, label="Synthetic → nearest training row")
    ax.set_xlabel("Euclidean distance in standardised feature space")
    ax.set_ylabel("Density")
    ax.legend()
    fig.suptitle(title)
    return _save(fig, path)


def plot_latent_space(mu_2d: np.ndarray, y: np.ndarray, axis_label: str, title: str, path: Path | None = None, seed: int = 0) -> plt.Figure:
    rng = np.random.default_rng(seed)
    normal_idx = rng.choice(np.flatnonzero(y == 0), min(8000, int((y == 0).sum())), replace=False)
    fraud_idx = np.flatnonzero(y == 1)
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    ax.scatter(mu_2d[normal_idx, 0], mu_2d[normal_idx, 1], s=4, color=NORMAL_COLOR, alpha=0.35, linewidths=0, label=f"Normal (sample of {len(normal_idx):,})")
    ax.scatter(mu_2d[fraud_idx, 0], mu_2d[fraud_idx, 1], s=22, color=FRAUD_COLOR, edgecolor=SURFACE, linewidth=0.6, label=f"Fraud ({len(fraud_idx)})")
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.legend(markerscale=2)
    ax.set_title("Coloured by class only after training (labels never seen by the model)", fontsize=9)
    fig.suptitle(title)
    return _save(fig, path)


def plot_kl_per_dim(kl: np.ndarray, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.bar([f"z{i + 1}" for i in range(len(kl))], kl, color=NORMAL_COLOR, width=0.6)
    ax.axhline(0.01, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title("Dimensions above the dotted line (0.01 nats) carry information", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.suptitle(title)
    return _save(fig, path)


# ----------------------------------------------------------------------------- E1 vs E2 comparison and analysis
E1_COLOR, E2_COLOR = NORMAL_COLOR, NEUTRAL


def plot_training_comparison(h1: pd.DataFrame, h2: pd.DataFrame, best1: int, best2: int, title: str, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, key, label in ((axes[0], "train_recon", "Training reconstruction term"), (axes[1], "train_kl", "Training KL divergence"),
                           (axes[2], "val_recon_mse_det", "Validation reconstruction MSE (z = μ)")):
        for h, best, color, name in ((h1, best1, E1_COLOR, "E1 normal-only"), (h2, best2, E2_COLOR, "E2 unsupervised")):
            ax.plot(h["epoch"], h[key], color=color, linewidth=2, label=f"{name} (best epoch {best})")
            ax.axvline(best, color=color, linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        lo, hi = np.percentile(np.r_[h1[key].iloc[10:], h2[key].iloc[10:]], [0, 100])
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
    axes[0].legend(loc="upper right", fontsize=7)
    axes[2].annotate("E2's validation set also contains frauds", (0.02, 0.04), xycoords="axes fraction", fontsize=7, color=TEXT_SECONDARY)
    fig.suptitle(title)
    return _save(fig, path)


def plot_score_distributions_pair(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, tau1: float, tau2: float, title: str, path: Path | None = None) -> plt.Figure:
    lo = max(min(s1.min(), s2.min()), 1e-3)
    bins = np.logspace(np.log10(lo), np.log10(max(s1.max(), s2.max())), 70)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), sharex=True, sharey=True, layout="constrained")
    for ax, s, tau, name in ((axes[0], s1, tau1, "E1 normal-only"), (axes[1], s2, tau2, "E2 unsupervised")):
        for k, color, label in ((0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")):
            ax.hist(s[y == k], bins=bins, density=True, histtype="step", linewidth=2, color=color, label=label)
            ax.axvline(np.median(s[y == k]), color=color, linewidth=0.8, linestyle=":")
        ax.axvline(tau, color=TEXT, linewidth=1.5, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{name}: median fraud/normal error ratio {np.median(s[y == 1]) / np.median(s[y == 0]):.1f}×", fontsize=10)
        ax.set_xlabel("Reconstruction error (log; dashed = τ, dotted = class medians)")
    axes[0].set_ylabel("Density within class")
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_metric_comparison(per_seed: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    metrics = [("average_precision", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    width = 0.36
    for j, (exp, color, label) in enumerate((("e1", E1_COLOR, "E1 normal-only"), ("e2", E2_COLOR, "E2 unsupervised"))):
        sub = per_seed[per_seed["experiment"] == exp]
        x = np.arange(len(metrics)) + (j - 0.5) * width
        means = [sub[m].mean() for m, _ in metrics]
        ax.bar(x, means, width=width * 0.92, color=color, alpha=0.4, label=f"{label} (mean of 5 seeds)")
        for i, (m, _) in enumerate(metrics):
            ax.scatter(np.full(len(sub), x[i]), sub[m], color=color, s=14, zorder=3)
            ax.annotate(f"{means[i]:.3f}", (x[i], max(sub[m])), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(np.arange(len(metrics)), [l for _, l in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test-set value (dots = individual seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_latent_pair(mu1_2d: np.ndarray, mu2_2d: np.ndarray, y: np.ndarray, title: str, path: Path | None = None, seed: int = 0) -> plt.Figure:
    rng = np.random.default_rng(seed)
    normal_idx = rng.choice(np.flatnonzero(y == 0), min(8000, int((y == 0).sum())), replace=False)
    fraud_idx = np.flatnonzero(y == 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    for ax, m2, name in ((axes[0], mu1_2d, "E1 normal-only"), (axes[1], mu2_2d, "E2 unsupervised")):
        ax.scatter(m2[normal_idx, 0], m2[normal_idx, 1], s=4, color=NORMAL_COLOR, alpha=0.3, linewidths=0, label="Normal (sample)")
        ax.scatter(m2[fraud_idx, 0], m2[fraud_idx, 1], s=22, color=FRAUD_COLOR, edgecolor=SURFACE, linewidth=0.6, label="Fraud")
        ax.set_title(name)
        ax.set_xlabel("PCA of μ, component 1")
        ax.set_ylabel("PCA of μ, component 2")
        ax.legend(markerscale=2, loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_latent_dims(mu: np.ndarray, y: np.ndarray, title: str, path: Path | None = None) -> plt.Figure:
    d = mu.shape[1]
    fig, axes = plt.subplots(2, (d + 1) // 2, figsize=(16, 6), layout="constrained")
    for k, ax in enumerate(axes.flat[:d]):
        lo, hi = np.percentile(mu[:, k], [0.2, 99.8])
        lo, hi = min(lo, np.percentile(mu[y == 1, k], 2)), max(hi, np.percentile(mu[y == 1, k], 98))
        bins = np.linspace(lo, hi, 50)
        for cls, color, label in ((0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")):
            ax.hist(np.clip(mu[y == cls, k], lo, hi), bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=label)
        ax.set_title(f"z{k + 1}")
        ax.set_yticks([])
        ax.grid(visible=False)
    for ax in axes.flat[d:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    return _save(fig, path)


def plot_per_feature_errors(t1: pd.DataFrame, t2: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    order = t1.sort_values("fraud_mean_sq_error", ascending=False).index
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, layout="constrained")
    x = np.arange(len(order))
    for ax, t, name in ((axes[0], t1, "E1 normal-only"), (axes[1], t2, "E2 unsupervised")):
        t = t.reindex(order)
        ax.bar(x - 0.2, t["normal_mean_sq_error"], width=0.4, color=NORMAL_COLOR, label="Normal")
        ax.bar(x + 0.2, t["fraud_mean_sq_error"], width=0.4, color=FRAUD_COLOR, label="Fraud")
        ax.set_yscale("log")
        ax.set_ylabel("Mean squared error (log)")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    axes[1].set_xticks(x, order, rotation=90)
    fig.suptitle(title)
    return _save(fig, path)


def plot_reconstruction_scatter(X: np.ndarray, R: np.ndarray, y: np.ndarray, features: list[int], names: list[str], title: str,
                                path: Path | None = None, seed: int = 0) -> plt.Figure:
    rng = np.random.default_rng(seed)
    idx_n = rng.choice(np.flatnonzero(y == 0), 4000, replace=False)
    idx_f = np.flatnonzero(y == 1)
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.6), layout="constrained")
    for ax, j in zip(np.atleast_1d(axes), features):
        lo, hi = np.percentile(np.r_[X[idx_n, j], X[idx_f, j]], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax.scatter(X[idx_n, j], R[idx_n, j], s=4, color=NORMAL_COLOR, alpha=0.35, linewidths=0, label="Normal")
        ax.scatter(X[idx_f, j], R[idx_f, j], s=20, color=FRAUD_COLOR, edgecolor=SURFACE, linewidth=0.5, label="Fraud")
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(names[j])
        ax.set_xlabel("Actual (standardised)")
        ax.set_ylabel("Reconstructed")
    np.atleast_1d(axes)[0].legend(markerscale=2)
    fig.suptitle(title)
    return _save(fig, path)


def plot_precision_vs_prevalence(prevalences: np.ndarray, precision: np.ndarray, actual: float, tpr: float, fpr: float, title: str,
                                 path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(100 * prevalences, precision, color=NORMAL_COLOR, linewidth=2)
    for p in (actual, 0.01, 0.1):
        v = tpr * p / (tpr * p + fpr * (1 - p))
        ax.scatter([100 * p], [v], color=TEXT, s=30, zorder=3)
        ax.annotate(f"fraud rate {100 * p:g}% → precision {v:.2f}", (100 * p, v), xytext=(6, -10), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Fraud prevalence (%) (log scale)")
    ax.set_ylabel("Expected precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Same detector everywhere: TPR {tpr:.2f}, FPR {fpr:.4f} (E1 seed 0 at τ)", fontsize=9)
    fig.suptitle(title)
    return _save(fig, path)
