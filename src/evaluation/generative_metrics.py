"""Real-vs-synthetic comparison metrics. 'Real' = held-out normal test transactions (never used for any decision)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

QUANTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)


def moment_table(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"real_mean": real.mean(), "synthetic_mean": synth.mean(), "real_std": real.std(), "synthetic_std": synth.std()})
    t["mean_diff_in_real_std"] = (t["synthetic_mean"] - t["real_mean"]) / t["real_std"]
    t["std_ratio"] = t["synthetic_std"] / t["real_std"]
    return t


def distance_table(real_z: np.ndarray, synth_z: np.ndarray, names: list[str]) -> pd.DataFrame:
    """KS statistic and Wasserstein-1 distance per feature, in standardised model space (comparable across features)."""
    rows = []
    for j, name in enumerate(names):
        ks = ks_2samp(real_z[:, j], synth_z[:, j])
        rows.append({"feature": name, "ks_statistic": ks.statistic, "wasserstein_std_units": wasserstein_distance(real_z[:, j], synth_z[:, j])})
    return pd.DataFrame(rows).set_index("feature")


def quantile_table(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rq, sq = real.quantile(list(QUANTILES)), synth.quantile(list(QUANTILES))
    out = {}
    for col in real.columns:
        for q in QUANTILES:
            out.setdefault(col, {})[f"real_q{q}"] = rq.loc[q, col]
            out[col][f"synthetic_q{q}"] = sq.loc[q, col]
        out[col]["central_98pct_width_ratio"] = (sq.loc[0.99, col] - sq.loc[0.01, col]) / (rq.loc[0.99, col] - rq.loc[0.01, col])
        out[col]["tail_99_9pct_width_ratio"] = (sq.loc[0.999, col] - sq.loc[0.001, col]) / (rq.loc[0.999, col] - rq.loc[0.001, col])
    return pd.DataFrame(out).T


def correlation_comparison(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    cr, cs = real.corr().to_numpy(), synth.corr().to_numpy()
    diff = cs - cr
    off = ~np.eye(len(cr), dtype=bool)
    amount_pairs = {f"Amount~{v}": {"real": float(real["Amount"].corr(real[v])), "synthetic": float(synth["Amount"].corr(synth[v]))}
                    for v in ("V2", "V5", "V7", "V20", "V1")}
    return {
        "frobenius_norm_of_difference": float(np.linalg.norm(diff)),
        "mean_abs_offdiag_difference": float(np.abs(diff[off]).mean()),
        "max_abs_offdiag_difference": float(np.abs(diff[off]).max()),
        "mean_abs_offdiag_real": float(np.abs(cr[off]).mean()),
        "mean_abs_offdiag_synthetic": float(np.abs(cs[off]).mean()),
        "amount_correlations": amount_pairs,
        "real_matrix": cr, "synthetic_matrix": cs,
    }


def pca_projection(real_z: np.ndarray, synth_z: np.ndarray) -> dict:
    pca = PCA(n_components=2, random_state=0).fit(real_z)
    return {"real": pca.transform(real_z), "synthetic": pca.transform(synth_z),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def classifier_two_sample_test(real_z: np.ndarray, synth_z: np.ndarray, n: int, seed: int) -> dict:
    """Can a classifier tell real from synthetic? 5-fold CV ROC-AUC; 0.5 = indistinguishable."""
    rng = np.random.default_rng(seed)
    r = real_z[rng.choice(len(real_z), min(n, len(real_z)), replace=False)]
    s = synth_z[rng.choice(len(synth_z), min(n, len(synth_z)), replace=False)]
    X = np.vstack([r, s])
    y = np.r_[np.zeros(len(r)), np.ones(len(s))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {"samples_per_class": int(len(r))}
    for name, clf in {"logistic_regression": LogisticRegression(max_iter=2000),
                      "gradient_boosting": HistGradientBoostingClassifier(random_state=seed)}.items():
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        out[name] = {"roc_auc_mean": float(auc.mean()), "roc_auc_std": float(auc.std())}
    return out


def _nearest_distance(queries: np.ndarray, reference: np.ndarray, chunk: int = 512) -> np.ndarray:
    ref_sq = (reference ** 2).sum(1)
    out = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        d2 = (q ** 2).sum(1)[:, None] + ref_sq[None, :] - 2 * q @ reference.T
        out[start:start + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def memorisation_check(train_z: np.ndarray, holdout_z: np.ndarray, synth_z: np.ndarray, n: int, seed: int) -> dict:
    """Distance to the closest training record: synthetic vs real held-out rows (standardised space)."""
    rng = np.random.default_rng(seed)
    s = synth_z[rng.choice(len(synth_z), min(n, len(synth_z)), replace=False)]
    h = holdout_z[rng.choice(len(holdout_z), min(n, len(holdout_z)), replace=False)]
    ds, dh = _nearest_distance(s, train_z), _nearest_distance(h, train_z)
    return {
        "samples": int(len(s)),
        "synthetic_to_train": {"median": float(np.median(ds)), "p05": float(np.percentile(ds, 5)), "min": float(ds.min())},
        "holdout_to_train": {"median": float(np.median(dh)), "p05": float(np.percentile(dh, 5)), "min": float(dh.min())},
        "synthetic_exact_copies": int(np.sum(ds < 1e-6)),
        "share_synthetic_closer_than_holdout_p05": float(np.mean(ds < np.percentile(dh, 5))),
        "distances": {"synthetic": ds, "holdout": dh},
    }


def validity(df: pd.DataFrame, real: pd.DataFrame) -> dict:
    finite = np.isfinite(df.to_numpy()).all(axis=1)
    lo, hi = real.min(), real.max()
    within = ((df >= lo) & (df <= hi)).all(axis=1)
    return {"rows": len(df), "finite_pct": 100 * finite.mean(), "amount_non_negative_pct": 100 * (df["Amount"] >= 0).mean(),
            "amount_two_decimals_pct": 100 * np.isclose(df["Amount"], df["Amount"].round(2)).mean(),
            "within_real_observed_range_pct": 100 * within.mean(),
            "amount_zero_pct_synthetic": 100 * (df["Amount"] == 0).mean(), "amount_zero_pct_real": 100 * (real["Amount"] == 0).mean()}
