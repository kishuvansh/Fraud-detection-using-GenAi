"""Anomaly-detection metrics for imbalanced data (fraud = positive class)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_metrics(y: np.ndarray, scores: np.ndarray, tau: float) -> dict:
    pred = scores > tau
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(tau), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "tpr": recall, "fpr": fp / (fp + tn) if fp + tn else 0.0, "specificity": tn / (fp + tn) if fp + tn else 0.0,
        "flagged": tp + fp, "flagged_pct": 100 * (tp + fp) / len(y),
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "prevalence": float(np.mean(y)),
    }


def recall_at_budget(y: np.ndarray, scores: np.ndarray, budgets: list[int]) -> dict:
    order = np.argsort(-scores, kind="stable")
    out = {}
    for k in budgets:
        top = y[order[:k]]
        out[str(k)] = {"frauds_caught": int(top.sum()), "recall": float(top.sum() / y.sum()), "precision": float(top.mean())}
    return out


def sensitivity_table(y_test: np.ndarray, test_scores: np.ndarray, val_scores: np.ndarray, percentiles: list[float],
                      primary: float) -> pd.DataFrame:
    """Thresholds come ONLY from validation percentiles; each is then applied to the test set."""
    rows = []
    for q in percentiles:
        tau = float(np.percentile(val_scores, q))
        m = threshold_metrics(y_test, test_scores, tau)
        rows.append({"validation_percentile": q, "role": "primary" if q == primary else "sensitivity", **m})
    return pd.DataFrame(rows)


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, tau: float, n_resamples: int, seed: int) -> dict:
    """95% percentile intervals from class-stratified resampling of the test set."""
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("average_precision", "roc_auc", "precision", "recall", "f1", "fpr")}
    for _ in range(n_resamples):
        idx = _stratified_indices(y, rng)
        yy, ss = y[idx], scores[idx]
        draws["average_precision"].append(average_precision_score(yy, ss))
        draws["roc_auc"].append(roc_auc_score(yy, ss))
        m = threshold_metrics(yy, ss, tau)
        for k in ("precision", "recall", "f1", "fpr"):
            draws[k].append(m[k])
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in draws.items()}


def error_profile(y: np.ndarray, predicted: np.ndarray, scores: np.ndarray, Z: np.ndarray, amount: np.ndarray) -> pd.DataFrame:
    """Median profile of TP / FN / FP / TN rows. Z = standardised model inputs (used for max |z|)."""
    group = np.select([(y == 1) & (predicted == 1), (y == 1) & (predicted == 0), (y == 0) & (predicted == 1)], ["TP", "FN", "FP"], default="TN")
    t = pd.DataFrame({"group": group, "score": scores, "max_abs_z": np.abs(Z).max(axis=1), "amount": amount})
    out = t.groupby("group").agg(rows=("score", "size"), median_score=("score", "median"), median_max_abs_z=("max_abs_z", "median"),
                                 share_any_abs_z_gt_5=("max_abs_z", lambda s: float(np.mean(s > 5))), median_amount=("amount", "median"))
    return out.reindex(["TP", "FN", "FP", "TN"])


def paired_bootstrap_ap_difference(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_resamples: int,
                                   seed: int) -> dict:
    """AP(a) - AP(b) on the same resampled test rows."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = _stratified_indices(y, rng)
        diffs.append(average_precision_score(y[idx], scores_a[idx]) - average_precision_score(y[idx], scores_b[idx]))
    diffs = np.asarray(diffs)
    return {"observed": float(average_precision_score(y, scores_a) - average_precision_score(y, scores_b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "share_of_resamples_a_better": float(np.mean(diffs > 0))}
