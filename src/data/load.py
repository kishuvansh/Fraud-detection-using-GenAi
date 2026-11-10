"""Read-only loading of creditcard.csv and separation of model inputs from the label."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import verify_sha256

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
TIME_COL = "Time"
AMOUNT_COL = "Amount"
TARGET_COL = "Class"
EXPECTED_COLUMNS = [TIME_COL, *PCA_FEATURES, AMOUNT_COL, TARGET_COL]

# Raw columns the model is allowed to see (before transformation). Time and Class are excluded.
RAW_INPUT_COLUMNS = [*PCA_FEATURES, AMOUNT_COL]


def raw_data_path(root: Path, cfg: dict) -> Path:
    return root / cfg["data"]["raw_path"]


def load_raw(root: Path, cfg: dict, verify: bool = True) -> pd.DataFrame:
    """Load the full dataset. The file is only read, never written."""
    path = raw_data_path(root, cfg)
    if verify:
        verify_sha256(path, cfg["data"]["sha256"])
    dtypes = {col: "float64" for col in EXPECTED_COLUMNS if col != TARGET_COL}
    dtypes[TARGET_COL] = "int64"  # stored as quoted "0"/"1" in the CSV
    df = pd.read_csv(path, dtype=dtypes)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != cfg["data"]["expected_rows"]:
        raise ValueError(f"Expected {cfg['data']['expected_rows']} rows, found {len(df)}")
    if not set(np.unique(df[TARGET_COL])) <= {0, 1}:
        raise ValueError("Class must contain only 0 and 1")
    return df


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (model-input columns, labels) as separate objects. Class never appears in X."""
    X = df[RAW_INPUT_COLUMNS].copy()
    y = df[TARGET_COL].to_numpy(dtype=np.int64)
    assert TARGET_COL not in X.columns and TIME_COL not in X.columns
    return X, y
