"""Descriptive statistics for exploratory data analysis.

Statistics that use the Class label are descriptive only. They do not drive any modelling
decision: all features are kept and every modelling choice was fixed in PROJECT_PLAN.md beforehand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import AMOUNT_COL, PCA_FEATURES, RAW_INPUT_COLUMNS, TARGET_COL, TIME_COL

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def quality_report(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=np.number)
    exact_dup = df.duplicated()
    vector_counts = df.groupby(RAW_INPUT_COLUMNS, sort=False).size()
    labels_per_vector = df.groupby(RAW_INPUT_COLUMNS, sort=False)[TARGET_COL].nunique()
    top_freq_pct = {c: round(100 * df[c].value_counts().iloc[0] / len(df), 4) for c in df.columns}
    return {
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "infinite_total": int(np.isinf(numeric.to_numpy()).sum()),
        "non_numeric_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
        "exact_duplicate_rows": int(exact_dup.sum()),
        "exact_duplicate_rows_by_class": {int(k): int(v) for k, v in df.loc[exact_dup, TARGET_COL].value_counts().items()},
        "rows_repeating_a_vector_ignoring_time": int(df.duplicated(subset=RAW_INPUT_COLUMNS + [TARGET_COL]).sum()),
        "distinct_input_vectors": int(len(vector_counts)),
        "vectors_occurring_more_than_once": int((vector_counts > 1).sum()),
        "max_copies_of_one_vector": int(vector_counts.max()),
        "vectors_with_conflicting_labels": int((labels_per_vector > 1).sum()),
        "unique_values_per_column": {c: int(v) for c, v in df.nunique().items()},
        "most_frequent_value_share_pct": top_freq_pct,
        "constant_columns": [c for c in df.columns if df[c].nunique() <= 1],
        "near_constant_columns_gt_99pct": [c for c, p in top_freq_pct.items() if p > 99 and c != TARGET_COL],
        "negative_amounts": int((df[AMOUNT_COL] < 0).sum()),
        "zero_amounts": int((df[AMOUNT_COL] == 0).sum()),
        "zero_amounts_by_class": {int(k): int(v) for k, v in df.loc[df[AMOUNT_COL] == 0, TARGET_COL].value_counts().items()},
        "negative_time": int((df[TIME_COL] < 0).sum()),
        "time_is_sorted": bool(df[TIME_COL].is_monotonic_increasing),
        "class_values": sorted(int(v) for v in df[TARGET_COL].unique()),
    }


def class_balance(df: pd.DataFrame) -> dict:
    counts = df[TARGET_COL].value_counts().sort_index()
    normal, fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    return {
        "normal": normal,
        "fraud": fraud,
        "total": normal + fraud,
        "fraud_pct": 100 * fraud / (normal + fraud),
        "normal_pct": 100 * normal / (normal + fraud),
        "normal_per_fraud": normal / fraud,
    }


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TIME_COL, *PCA_FEATURES, AMOUNT_COL]
    table = df[cols].describe(percentiles=[0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]).T
    table["skew"] = df[cols].skew()
    table["excess_kurtosis"] = df[cols].kurt()
    table["n_unique"] = df[cols].nunique()
    return table


def amount_summary(df: pd.DataFrame) -> dict:
    amount = df[AMOUNT_COL]
    by_class = df.groupby(TARGET_COL)[AMOUNT_COL].describe(percentiles=[0.25, 0.5, 0.75, 0.99])
    return {
        "overall": amount.describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_dict(),
        "by_class": {int(k): v for k, v in by_class.to_dict(orient="index").items()},
        "skew_raw": float(amount.skew()),
        "skew_log1p": float(np.log1p(amount).skew()),
        "excess_kurtosis_raw": float(amount.kurt()),
        "max_decimal_places": int(df[AMOUNT_COL].astype(str).str.split(".").str[1].str.len().max()),
    }


def time_summary(df: pd.DataFrame) -> dict:
    hours_since_start = (df[TIME_COL] // SECONDS_PER_HOUR).astype(int)
    hour_of_cycle = ((df[TIME_COL] % SECONDS_PER_DAY) // SECONDS_PER_HOUR).astype(int)
    per_cycle = pd.crosstab(hour_of_cycle, df[TARGET_COL])
    per_cycle["fraud_rate_pct"] = 100 * per_cycle[1] / per_cycle.sum(axis=1)
    day = np.where(df[TIME_COL] < SECONDS_PER_DAY, "day_1", "day_2")
    per_day = pd.crosstab(day, df[TARGET_COL])
    return {
        "min_seconds": float(df[TIME_COL].min()),
        "max_seconds": float(df[TIME_COL].max()),
        "span_hours": float(df[TIME_COL].max() / SECONDS_PER_HOUR),
        "unique_values": int(df[TIME_COL].nunique()),
        "is_sorted": bool(df[TIME_COL].is_monotonic_increasing),
        "hours_since_start_bins": int(hours_since_start.nunique()),
        "per_day": {d: {"normal": int(r[0]), "fraud": int(r[1]), "fraud_rate_pct": 100 * r[1] / (r[0] + r[1])}
                    for d, r in per_day.iterrows()},
        "per_hour_of_cycle": {int(h): {"normal": int(r[0]), "fraud": int(r[1]), "fraud_rate_pct": float(r["fraud_rate_pct"])}
                              for h, r in per_cycle.iterrows()},
        "corr_with_V_top": df[PCA_FEATURES].corrwith(df[TIME_COL]).abs().sort_values(ascending=False).head(5).round(4).to_dict(),
    }


def _max_offdiag_abs_corr(frame: pd.DataFrame) -> float:
    corr = frame.corr().to_numpy()
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


def pca_feature_checks(df: pd.DataFrame) -> dict:
    stds = df[PCA_FEATURES].std()
    return {
        "max_abs_mean": float(df[PCA_FEATURES].mean().abs().max()),
        "std_min": float(stds.min()),
        "std_max": float(stds.max()),
        "std_ratio_max_over_min": float(stds.max() / stds.min()),
        "std_strictly_decreasing": bool((np.diff(stds.to_numpy()) < 0).all()),
        "max_abs_offdiag_corr_all_rows": _max_offdiag_abs_corr(df[PCA_FEATURES]),
        "max_abs_offdiag_corr_normal_rows": _max_offdiag_abs_corr(df.loc[df[TARGET_COL] == 0, PCA_FEATURES]),
        "max_abs_offdiag_corr_fraud_rows": _max_offdiag_abs_corr(df.loc[df[TARGET_COL] == 1, PCA_FEATURES]),
        "amount_corr_with_V_top": df[PCA_FEATURES].corrwith(df[AMOUNT_COL]).abs().sort_values(ascending=False).head(5).round(4).to_dict(),
        "excess_kurtosis_range": [float(df[PCA_FEATURES].kurt().min()), float(df[PCA_FEATURES].kurt().max())],
        "skew_range": [float(df[PCA_FEATURES].skew().min()), float(df[PCA_FEATURES].skew().max())],
    }


def class_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature normal vs fraud summary. smd = (mean_fraud - mean_normal) / std_normal."""
    cols = [*PCA_FEATURES, AMOUNT_COL, TIME_COL]
    normal = df.loc[df[TARGET_COL] == 0, cols]
    fraud = df.loc[df[TARGET_COL] == 1, cols]
    table = pd.DataFrame({
        "mean_normal": normal.mean(),
        "mean_fraud": fraud.mean(),
        "median_normal": normal.median(),
        "median_fraud": fraud.median(),
        "std_normal": normal.std(),
        "std_fraud": fraud.std(),
    })
    table["smd"] = (table["mean_fraud"] - table["mean_normal"]) / table["std_normal"]
    table["abs_smd"] = table["smd"].abs()
    return table.sort_values("abs_smd", ascending=False)


def extreme_value_report(df: pd.DataFrame, z: float = 5.0) -> dict:
    v = df[PCA_FEATURES]
    zscores = (v - v.mean()) / v.std()
    any_extreme = (zscores.abs() > z).any(axis=1)
    by_class = df.loc[any_extreme, TARGET_COL].value_counts()
    n_fraud = int((df[TARGET_COL] == 1).sum())
    return {
        "z_threshold": z,
        "rows_with_any_extreme_feature": int(any_extreme.sum()),
        "normal_rows": int(by_class.get(0, 0)),
        "fraud_rows": int(by_class.get(1, 0)),
        "share_of_frauds_pct": 100 * int(by_class.get(1, 0)) / n_fraud,
    }


def compute_eda_stats(df: pd.DataFrame) -> dict:
    return {
        "overview": dataset_overview(df),
        "quality": quality_report(df),
        "class_balance": class_balance(df),
        "amount": amount_summary(df),
        "time": time_summary(df),
        "pca_features": pca_feature_checks(df),
        "extreme_values": extreme_value_report(df),
    }
