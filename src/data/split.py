"""Reproducible, leakage-safe stratified group split (PROJECT_PLAN.md §3.3).

* Every row of the dataset lands in exactly one of train / val / test. No row is dropped.
* Rows with identical V1-V28 + Amount values form a group, and a group never spans two splits.
* Allocation is stratified by class: groups are shuffled within each class with a fixed seed and
  assigned to splits until each split's share of that class's rows is reached.
* Labels are used only for this partitioning, never as model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import RAW_INPUT_COLUMNS, TARGET_COL
from src.utils.io import load_json, save_json, sha256_array

SPLITS = ("train", "val", "test")


def assign_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id per row; rows with identical model-input values share an id (deterministic)."""
    return df.groupby(RAW_INPUT_COLUMNS, sort=True).ngroup().to_numpy(dtype=np.int64)


def stratified_group_split(groups: np.ndarray, y: np.ndarray, ratios: dict, seed: int) -> dict[str, np.ndarray]:
    if not np.isclose(sum(ratios[s] for s in SPLITS), 1.0):
        raise ValueError("Split ratios must sum to 1")
    labels_per_group = pd.Series(y).groupby(groups).nunique()
    if (labels_per_group > 1).any():
        raise ValueError("Some groups contain both classes; stratified group split is undefined")

    rng = np.random.default_rng(seed)
    split_of_group: dict[int, int] = {}
    for cls in np.unique(y):  # sorted: 0 then 1, so the RNG stream is deterministic
        cls_groups, sizes = np.unique(groups[y == cls], return_counts=True)
        order = rng.permutation(len(cls_groups))
        cls_groups, sizes = cls_groups[order], sizes[order]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        n_rows = sizes.sum()
        train_end = ratios["train"] * n_rows
        val_end = (ratios["train"] + ratios["val"]) * n_rows
        which = np.where(starts < train_end, 0, np.where(starts < val_end, 1, 2))
        split_of_group.update(zip(cls_groups.tolist(), which.tolist()))

    row_split = np.array([split_of_group[g] for g in groups.tolist()], dtype=np.int8)
    return {name: np.flatnonzero(row_split == i) for i, name in enumerate(SPLITS)}


def derived_partitions(splits: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    """Experiment-specific index sets (labels used only to partition)."""
    return {
        "e1_train": splits["train"][y[splits["train"]] == 0],          # primary VAE: normal only
        "e1_val_calibration": splits["val"][y[splits["val"]] == 0],    # threshold + early stopping (E1)
        "e2_train": splits["train"],                                  # unsupervised VAE: all train rows
        "e2_val": splits["val"],                                      # E2 calibration: all val rows, unlabelled
        "test": splits["test"],                                       # final evaluation, both classes
    }


def partition_table(splits: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    rows = []
    total = len(y)
    for name in SPLITS:
        idx = splits[name]
        n_fraud = int(y[idx].sum())
        rows.append({
            "split": name,
            "rows": len(idx),
            "normal": len(idx) - n_fraud,
            "fraud": n_fraud,
            "fraud_pct": 100 * n_fraud / len(idx),
            "share_of_dataset_pct": 100 * len(idx) / total,
        })
    rows.append({"split": "total", "rows": total, "normal": int((y == 0).sum()), "fraud": int(y.sum()),
                 "fraud_pct": 100 * y.mean(), "share_of_dataset_pct": 100.0})
    return pd.DataFrame(rows).set_index("split")


def experiment_table(splits: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    parts = derived_partitions(splits, y)
    usage = {
        "e1_train": "E1 training (normal only)",
        "e1_val_calibration": "E1 early stopping + threshold calibration (normal only)",
        "e2_train": "E2 training (all train rows, labels ignored)",
        "e2_val": "E2 early stopping + threshold calibration (all val rows, labels ignored)",
        "test": "Final evaluation of E1 and E2 (labels read once, after predictions are saved)",
    }
    rows = []
    for name, idx in parts.items():
        n_fraud = int(y[idx].sum())
        rows.append({"partition": name, "rows": len(idx), "normal": len(idx) - n_fraud, "fraud": n_fraud, "used_for": usage[name]})
    val_fraud = splits["val"][y[splits["val"]] == 1]
    train_fraud = splits["train"][y[splits["train"]] == 1]
    rows.append({"partition": "train_fraud", "rows": len(train_fraud), "normal": 0, "fraud": len(train_fraud),
                 "used_for": "Excluded from E1 training; part of E2 training (unlabelled)"})
    rows.append({"partition": "val_fraud", "rows": len(val_fraud), "normal": 0, "fraud": len(val_fraud),
                 "used_for": "Not used for any development decision"})
    return pd.DataFrame(rows).set_index("partition")


def save_splits(path: Path, splits: dict[str, np.ndarray], meta: dict) -> None:
    payload = {"meta": {**meta, "index_sha256": {s: sha256_array(splits[s]) for s in SPLITS}}}
    payload.update({s: splits[s].tolist() for s in SPLITS})
    save_json(payload, path, indent=None)


def load_splits(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    payload = load_json(path)
    splits = {s: np.asarray(payload[s], dtype=np.int64) for s in SPLITS}
    for s in SPLITS:
        if sha256_array(splits[s]) != payload["meta"]["index_sha256"][s]:
            raise ValueError(f"Split '{s}' in {path} does not match its stored fingerprint")
    return splits, payload["meta"]
