"""EDA figures. Each function returns a Matplotlib figure and optionally saves it as PNG.

Colours: normal = blue, fraud = orange (validated colour-blind-safe pair); correlations use a
blue-gray-red diverging scale. No chart uses two y-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from src.data.load import AMOUNT_COL, PCA_FEATURES, TARGET_COL, TIME_COL

NORMAL_COLOR = "#2a78d6"
FRAUD_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a32d2c"])
CLASS_LABELS = {0: "Normal", 1: "Fraud"}
CLASS_COLORS = {0: NORMAL_COLOR, 1: FRAUD_COLOR}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": TEXT,
        "font.size": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "semibold",
    })


def _save(fig: plt.Figure, path: Path | None) -> plt.Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


_thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")


def plot_class_counts(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = [CLASS_LABELS[k] for k in counts.index]
    colors = [CLASS_COLORS[k] for k in counts.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, log in zip(axes, (False, True)):
        bars = ax.bar(labels, counts.values, color=colors, width=0.55)
        ax.set_yscale("log" if log else "linear")
        ax.set_title("Log scale (shows the fraud bar)" if log else "Linear scale")
        ax.set_ylabel("Transactions")
        if not log:
            ax.yaxis.set_major_formatter(_thousands)
        ax.grid(axis="x", visible=False)
        for bar, n in zip(bars, counts.values):
            ax.annotate(f"{n:,}\n({100 * n / counts.sum():.3f}%)", (bar.get_x() + bar.get_width() / 2, n),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.margins(y=0.25)
    fig.suptitle("Normal vs fraud transaction counts")
    return _save(fig, path)


def plot_class_share(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    counts = df[TARGET_COL].value_counts().sort_index()
    share = 100 * counts / counts.sum()
    ratio = counts[0] / counts[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.8), layout="constrained")
    zoom_title = f"Zoom on the last 1% (99–100%): 1 fraud per {ratio:.0f} normal transactions"
    for ax, (lo, title) in zip(axes, ((0, "Full range (0–100%)"), (99, zoom_title))):
        left = 0.0
        for k in counts.index:
            ax.barh([0], [share[k]], left=left, color=CLASS_COLORS[k], height=0.6,
                    edgecolor=SURFACE, linewidth=2, label=f"{CLASS_LABELS[k]}: {counts[k]:,} ({share[k]:.3f}%)")
            left += share[k]
        ax.set_xlim(lo, 100)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[1].text(100 - share[1] / 2, 0, f"Fraud\n{share[1]:.3f}%", ha="center", va="center", fontsize=8, color=SURFACE)
    axes[1].set_xlabel("Share of all transactions (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    fig.suptitle("Class distribution (extreme imbalance)")
    return _save(fig, path)


def plot_amount_distribution(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    amount = df[AMOUNT_COL]

    ax = axes[0]
    ax.hist(amount, bins=200, color=NORMAL_COLOR, edgecolor=SURFACE, linewidth=0.3)
    ax.set_yscale("log")
    ax.set_title("Raw Amount (all rows, log y-axis)")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Transactions (log)")
    ax.xaxis.set_major_formatter(_thousands)

    ax = axes[1]
    bins = np.linspace(0, np.log1p(amount.max()), 70)
    for k in (0, 1):
        ax.hist(np.log1p(df.loc[df[TARGET_COL] == k, AMOUNT_COL]), bins=bins, density=True, histtype="step",
                linewidth=2, color=CLASS_COLORS[k], label=CLASS_LABELS[k])
    ax.set_title("log1p(Amount) by class (density)")
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[2]
    for k in (0, 1):
        vals = np.sort(df.loc[df[TARGET_COL] == k, AMOUNT_COL].to_numpy())
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(np.maximum(vals, 0.01), ecdf, color=CLASS_COLORS[k], linewidth=2, label=CLASS_LABELS[k])
    ax.set_xscale("log")
    ax.set_title("Cumulative distribution of Amount by class")
    ax.set_xlabel("Amount (log scale; zeros drawn at 0.01)")
    ax.set_ylabel("Share of class at or below amount")
    ax.legend()
    fig.suptitle("Transaction amount distribution")
    return _save(fig, path)


def plot_time_distribution(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    hours = df[TIME_COL] / 3600
    bins = np.arange(0, 49, 1)
    normal_counts, _ = np.histogram(hours[df[TARGET_COL] == 0], bins=bins)
    fraud_counts, _ = np.histogram(hours[df[TARGET_COL] == 1], bins=bins)
    rate = 100 * fraud_counts / (normal_counts + fraud_counts)
    overall = 100 * df[TARGET_COL].mean()
    centers = bins[:-1] + 0.5

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].bar(centers, normal_counts, width=0.9, color=NORMAL_COLOR)
    axes[0].set_title("Normal transactions per hour")
    axes[0].set_ylabel("Transactions")
    axes[0].yaxis.set_major_formatter(_thousands)
    axes[1].bar(centers, fraud_counts, width=0.9, color=FRAUD_COLOR)
    axes[1].set_title("Fraudulent transactions per hour")
    axes[1].set_ylabel("Transactions")
    axes[2].plot(centers, rate, color=FRAUD_COLOR, linewidth=2, marker="o", markersize=4)
    axes[2].axhline(overall, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axes[2].annotate(f"overall {overall:.3f}%", (47.5, overall), xytext=(0, 4), textcoords="offset points",
                     ha="right", fontsize=8, color=TEXT_SECONDARY)
    axes[2].set_title("Fraud rate per hour")
    axes[2].set_ylabel("Fraud rate (%)")
    axes[2].set_xlabel("Hours since the first transaction in the file (clock time of hour 0 is not documented)")
    axes[2].set_xticks(np.arange(0, 49, 4))
    for ax in axes:
        ax.axvline(24, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Transaction time distribution (Time = seconds since first transaction, ~48 h)")
    return _save(fig, path)


def plot_feature_scales(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    stds = df[PCA_FEATURES].std()
    kurt = df[PCA_FEATURES].kurt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), layout="constrained")
    axes[0].bar(PCA_FEATURES, stds.values, color=NORMAL_COLOR, width=0.7)
    axes[0].axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axes[0].set_title("Standard deviation per feature (dashed line = 1)")
    axes[0].set_ylabel("Std (all rows)")
    axes[1].bar(PCA_FEATURES, kurt.values, color=NORMAL_COLOR, width=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("Excess kurtosis per feature (log scale; 0 = Gaussian)")
    axes[1].set_ylabel("Excess kurtosis")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="x", visible=False)
    fig.suptitle("V1–V28 are centred but not unit-scaled, and many are heavy-tailed")
    return _save(fig, path)


def plot_feature_grid(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(7, 4, figsize=(15, 20), layout="constrained")
    for ax, col in zip(axes.flat, PCA_FEATURES):
        lo, hi = df[col].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        for k in (0, 1):
            vals = df.loc[df[TARGET_COL] == k, col]
            ax.hist(vals.clip(lo, hi), bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=CLASS_COLORS[k], label=CLASS_LABELS[k])
        ax.set_title(col)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        ax.grid(visible=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, fontsize=10)
    fig.suptitle("V1–V28 distributions by class (density; view limited to the 0.5–99.5th percentile,\n"
                 "values outside are piled into the edge bins)")
    return _save(fig, path)


def plot_selected_features(df: pd.DataFrame, features: list[str], path: Path | None = None) -> plt.Figure:
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4), layout="constrained")
    for ax, col in zip(np.atleast_1d(axes), features):
        data = [df.loc[df[TARGET_COL] == k, col].to_numpy() for k in (0, 1)]
        box = ax.boxplot(data, tick_labels=["Normal", "Fraud"], widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={"color": TEXT, "linewidth": 1.5})
        for patch, k in zip(box["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_title(col)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Selected features: normal vs fraud (boxes = IQR, whiskers = 1.5×IQR, outliers hidden)")
    return _save(fig, path)


def plot_class_comparison(comparison: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    table = comparison.sort_values("smd")
    fig, ax = plt.subplots(figsize=(8, 8.5), layout="constrained")
    ax.barh(table.index, table["smd"], color="#6f6e69", height=0.7)  # neutral: bars are features, not classes
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("Standardised mean difference = (mean fraud − mean normal) / std normal")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    for name, v in table["smd"].items():
        if abs(v) >= 2:
            ax.annotate(f"{v:+.1f}", (v, name), xytext=(3 if v > 0 else -3, 0), textcoords="offset points",
                        ha="left" if v > 0 else "right", va="center", fontsize=7, color=TEXT)
    ax.set_title("How far fraud means sit from normal means (right = higher in fraud, left = lower)")
    fig.suptitle("Normal vs fraud feature comparison")
    return _save(fig, path)


def plot_correlation_heatmaps(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    cols = [*PCA_FEATURES, AMOUNT_COL, TIME_COL]
    subsets = [("All rows", df), ("Normal rows only", df[df[TARGET_COL] == 0]), ("Fraud rows only", df[df[TARGET_COL] == 1])]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4), layout="constrained")
    image = None
    for ax, (title, frame) in zip(axes, subsets):
        corr = frame[cols].corr().to_numpy()
        image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(cols)), cols, fontsize=6.5)
        ax.grid(visible=False)
        ax.set_title(f"{title} (n = {len(frame):,})")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.suptitle("Correlation heatmaps: V1–V28 are uncorrelated over all rows (PCA), but not within each class")
    return _save(fig, path)
