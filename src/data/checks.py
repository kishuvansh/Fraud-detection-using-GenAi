"""Automated leakage and integrity checks for the split and preprocessing (PROJECT_PLAN.md §7.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import RAW_INPUT_COLUMNS, TARGET_COL, TIME_COL
from src.data.preprocess import FeatureTransformer, _engineer
from src.data.split import SPLITS, assign_groups, derived_partitions, stratified_group_split
from src.utils.io import sha256_array


def _result(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def run_leakage_checks(df: pd.DataFrame, splits: dict[str, np.ndarray], transformers: dict[str, FeatureTransformer],
                       cfg: dict) -> list[dict]:
    y = df[TARGET_COL].to_numpy()
    n = len(df)
    parts = derived_partitions(splits, y)
    results = []

    # 1. Partition integrity
    all_idx = np.concatenate([splits[s] for s in SPLITS])
    results.append(_result("splits_are_disjoint", len(all_idx) == len(np.unique(all_idx)),
                           f"{len(all_idx):,} assigned indices, {len(np.unique(all_idx)):,} unique"))
    results.append(_result("every_row_assigned_no_subset", len(all_idx) == n and np.array_equal(np.sort(all_idx), np.arange(n)),
                           f"{len(all_idx):,} of {n:,} rows assigned"))

    # 2. Duplicate-vector leakage
    groups = assign_groups(df)
    split_of_row = np.empty(n, dtype=np.int8)
    for i, s in enumerate(SPLITS):
        split_of_row[splits[s]] = i
    splits_per_group = pd.Series(split_of_row).groupby(groups).nunique()
    results.append(_result("no_group_spans_two_splits", (splits_per_group == 1).all(),
                           f"{int((splits_per_group > 1).sum())} of {len(splits_per_group):,} groups span >1 split"))
    row_hash = pd.util.hash_pandas_object(df[RAW_INPUT_COLUMNS], index=False).to_numpy()
    train_hashes = set(row_hash[splits["train"]].tolist())
    overlap_test = sum(h in train_hashes for h in row_hash[splits["test"]].tolist())
    overlap_val = sum(h in train_hashes for h in row_hash[splits["val"]].tolist())
    results.append(_result("no_identical_vector_in_train_and_val_or_test", overlap_test == 0 and overlap_val == 0,
                           f"identical-to-train rows: val {overlap_val}, test {overlap_test} (independent hash check)"))

    # 3. Stratification
    overall = 100 * y.mean()
    tol = cfg["split"]["stratify_tolerance_pp"]
    rates = {s: 100 * y[splits[s]].mean() for s in SPLITS}
    results.append(_result("class_rate_preserved_in_each_split", all(abs(r - overall) <= tol for r in rates.values()),
                           "fraud % " + ", ".join(f"{s} {r:.4f}" for s, r in rates.items()) + f" vs overall {overall:.4f} (tolerance ±{tol} pp)"))

    # 4. Partition roles
    results.append(_result("e1_training_set_is_normal_only", y[parts["e1_train"]].sum() == 0,
                           f"{len(parts['e1_train']):,} rows, {int(y[parts['e1_train']].sum())} fraud"))
    results.append(_result("e1_calibration_set_is_normal_only", y[parts["e1_val_calibration"]].sum() == 0,
                           f"{len(parts['e1_val_calibration']):,} rows, {int(y[parts['e1_val_calibration']].sum())} fraud"))
    test_fraud = int(y[splits["test"]].sum())
    results.append(_result("test_set_has_unseen_normal_and_fraud", test_fraud > 0 and test_fraud < len(splits["test"]),
                           f"test: {len(splits['test']) - test_fraud:,} normal, {test_fraud} fraud"))

    # 5. Label isolation
    for name, tf in transformers.items():
        bad = [f for f in tf.feature_names if f in (TARGET_COL, TIME_COL)]
        results.append(_result(f"{name}_model_inputs_exclude_class_and_time", not bad,
                               f"{len(tf.feature_names)} inputs: V1-V28, log1p_Amount" if not bad else f"found {bad}"))

    # 6. Scalers fitted on permitted training rows only
    fit_rows = {"e1": parts["e1_train"], "e2": parts["e2_train"]}
    for name, tf in transformers.items():
        feats = _engineer(df.iloc[fit_rows[name]])
        ok = (np.allclose(tf.mean_, feats.mean(axis=0), rtol=0, atol=1e-12)
              and np.allclose(tf.scale_, feats.std(axis=0), rtol=0, atol=1e-12)
              and tf.n_fit_rows == len(fit_rows[name]))
        results.append(_result(f"{name}_scaler_fitted_on_its_training_rows_only", ok,
                               f"fitted on {tf.n_fit_rows:,} rows ({tf.fitted_on}); stats recomputed and matched"))
        val_test = _engineer(df.iloc[np.concatenate([splits["val"], splits["test"]])])
        differs = not np.allclose(tf.mean_, val_test.mean(axis=0), rtol=0, atol=1e-9)
        results.append(_result(f"{name}_scaler_differs_from_val_test_statistics", differs,
                               "fitted mean differs from val+test mean (sanity check that val/test were not used)"))

    # 7. Transform behaviour
    e1 = transformers["e1"]
    z_train = e1.transform(df.iloc[parts["e1_train"]])
    results.append(_result("e1_train_is_standardised", np.allclose(z_train.mean(0), 0, atol=1e-9) and np.allclose(z_train.std(0), 1, atol=1e-9),
                           f"max |mean| {np.abs(z_train.mean(0)).max():.2e}, max |std-1| {np.abs(z_train.std(0) - 1).max():.2e}"))
    sample = df.iloc[splits["test"]]
    back = e1.inverse_transform(e1.transform(sample), round_amount=False)
    v_err = np.abs(back[RAW_INPUT_COLUMNS[:-1]].to_numpy() - sample[RAW_INPUT_COLUMNS[:-1]].to_numpy()).max()
    a_err = np.abs(back["Amount"].to_numpy() - sample["Amount"].to_numpy()).max()
    results.append(_result("inverse_transform_round_trip", v_err < 1e-9 and a_err < 1e-6,
                           f"max abs error on test rows: V {v_err:.1e}, Amount {a_err:.1e}"))

    # 8. Reproducibility
    again = stratified_group_split(groups, y, cfg["split"]["ratios"], cfg["split"]["seed"])
    same = all(sha256_array(again[s]) == sha256_array(splits[s]) for s in SPLITS)
    results.append(_result("split_is_reproducible_with_same_seed", same, f"re-ran split with seed {cfg['split']['seed']}; fingerprints identical"))
    return results


def assert_all_passed(results: list[dict]) -> None:
    failed = [r for r in results if not r["passed"]]
    if failed:
        raise AssertionError("Leakage checks failed: " + "; ".join(f"{r['check']}: {r['detail']}" for r in failed))
