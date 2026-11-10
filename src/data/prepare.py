"""End-to-end data preparation (split + scalers + checks) and loaders for later phases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.checks import assert_all_passed, run_leakage_checks
from src.data.load import TARGET_COL, load_raw, raw_data_path
from src.data.preprocess import FeatureTransformer
from src.data.split import (assign_groups, derived_partitions, experiment_table, load_splits, partition_table,
                            save_splits, stratified_group_split)
from src.utils.io import save_json, sha256_file

SPLITS_FILE = "splits.json"


def processed_dir(root: Path, cfg: dict) -> Path:
    return root / cfg["paths"]["processed_dir"]


def fit_transformers(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict[str, FeatureTransformer]:
    y = df[TARGET_COL].to_numpy()
    parts = derived_partitions(splits, y)
    return {
        "e1": FeatureTransformer().fit(df.iloc[parts["e1_train"]], fitted_on="e1_train (normal training rows)"),
        "e2": FeatureTransformer().fit(df.iloc[parts["e2_train"]], fitted_on="e2_train (all training rows, labels ignored)"),
    }


def prepare_data(root: Path, cfg: dict) -> dict:
    """Split, fit scalers, run leakage checks, and save everything under data/processed and outputs/metrics."""
    raw_path = raw_data_path(root, cfg)
    hash_before = sha256_file(raw_path)
    df = load_raw(root, cfg)
    y = df[TARGET_COL].to_numpy()

    groups = assign_groups(df)
    splits = stratified_group_split(groups, y, cfg["split"]["ratios"], cfg["split"]["seed"])
    transformers = fit_transformers(df, splits)
    checks = run_leakage_checks(df, splits, transformers, cfg)
    hash_after = sha256_file(raw_path)
    checks.append({"check": "raw_dataset_unchanged", "passed": hash_before == hash_after == cfg["data"]["sha256"],
                   "detail": f"SHA-256 before and after preparation: {hash_after[:16]}…"})
    assert_all_passed(checks)

    out = processed_dir(root, cfg)
    meta = {"seed": cfg["split"]["seed"], "ratios": cfg["split"]["ratios"], "method": "stratified group split",
            "group_on": "V1-V28 + Amount", "n_groups": int(groups.max() + 1), "dataset_sha256": cfg["data"]["sha256"]}
    save_splits(out / SPLITS_FILE, splits, meta)
    for name, tf in transformers.items():
        tf.save(out / f"transformer_{name}.json")
    partitions = partition_table(splits, y)
    experiments = experiment_table(splits, y)
    partitions.to_csv(out / "split_summary.csv")
    experiments.to_csv(out / "experiment_partitions.csv")
    metrics = root / cfg["paths"]["metrics_dir"]
    save_json(checks, metrics / "leakage_checks.json")
    save_json({"splits": partitions.reset_index().to_dict(orient="records"),
               "experiment_partitions": experiments.reset_index().to_dict(orient="records"), "meta": meta},
              metrics / "partition_stats.json")
    return {"df": df, "splits": splits, "transformers": transformers, "checks": checks,
            "partitions": partitions, "experiments": experiments, "meta": meta}


def load_experiment_inputs(root: Path, cfg: dict, experiment: str, dtype=np.float32) -> dict:
    """Model inputs for one experiment ('e1' or 'e2'). Returns features only; no labels."""
    if experiment not in ("e1", "e2"):
        raise ValueError("experiment must be 'e1' or 'e2'")
    df = load_raw(root, cfg)
    splits, _ = load_splits(processed_dir(root, cfg) / SPLITS_FILE)
    parts = derived_partitions(splits, df[TARGET_COL].to_numpy())
    tf = FeatureTransformer.load(processed_dir(root, cfg) / f"transformer_{experiment}.json")
    train_key, val_key = ("e1_train", "e1_val_calibration") if experiment == "e1" else ("e2_train", "e2_val")
    return {
        "X_train": tf.transform(df.iloc[parts[train_key]], dtype=dtype),
        "X_val": tf.transform(df.iloc[parts[val_key]], dtype=dtype),
        "X_test": tf.transform(df.iloc[parts["test"]], dtype=dtype),
        "test_index": parts["test"],
        "transformer": tf,
        "feature_names": tf.feature_names,
    }


def load_test_labels(root: Path, cfg: dict) -> np.ndarray:
    """Test labels, kept separate from model inputs. Call only after test predictions are saved."""
    df = load_raw(root, cfg)
    splits, _ = load_splits(processed_dir(root, cfg) / SPLITS_FILE)
    return df[TARGET_COL].to_numpy()[splits["test"]]
