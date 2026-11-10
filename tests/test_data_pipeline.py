"""Data-pipeline tests. Run with `python tests/test_data_pipeline.py` (pytest also works if installed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.checks import run_leakage_checks  # noqa: E402
from src.data.load import PCA_FEATURES, TARGET_COL, load_raw, separate_features_target  # noqa: E402
from src.data.prepare import fit_transformers, load_experiment_inputs, processed_dir  # noqa: E402
from src.data.preprocess import FeatureTransformer  # noqa: E402
from src.data.split import assign_groups, load_splits, stratified_group_split  # noqa: E402
from src.utils.io import load_config  # noqa: E402

CFG = load_config(ROOT)
DF = load_raw(ROOT, CFG)


def test_features_and_target_are_separated():
    X, y = separate_features_target(DF)
    assert TARGET_COL not in X.columns and "Time" not in X.columns
    assert list(X.columns) == [*PCA_FEATURES, "Amount"]
    assert len(X) == len(y) == 284_807 and set(np.unique(y)) == {0, 1}


def test_split_on_toy_data_keeps_groups_together():
    toy = pd.DataFrame({c: [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] for c in [*PCA_FEATURES, "Amount"]})
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    splits = stratified_group_split(assign_groups(toy), y, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    where = {i: s for s, idx in splits.items() for i in idx}
    assert where[0] == where[1], "duplicate rows must share a split"
    assert sorted(where) == list(range(10))


def test_saved_splits_match_a_fresh_split():
    saved, meta = load_splits(processed_dir(ROOT, CFG) / "splits.json")
    fresh = stratified_group_split(assign_groups(DF), DF[TARGET_COL].to_numpy(), CFG["split"]["ratios"], CFG["split"]["seed"])
    assert meta["dataset_sha256"] == CFG["data"]["sha256"]
    for s in saved:
        assert np.array_equal(saved[s], fresh[s])


def test_all_leakage_checks_pass():
    splits, _ = load_splits(processed_dir(ROOT, CFG) / "splits.json")
    transformers = {name: FeatureTransformer.load(processed_dir(ROOT, CFG) / f"transformer_{name}.json") for name in ("e1", "e2")}
    failed = [r for r in run_leakage_checks(DF, splits, transformers, CFG) if not r["passed"]]
    assert not failed, failed


def test_saved_transformers_equal_refitted_ones():
    splits, _ = load_splits(processed_dir(ROOT, CFG) / "splits.json")
    refit = fit_transformers(DF, splits)
    for name, tf in refit.items():
        saved = FeatureTransformer.load(processed_dir(ROOT, CFG) / f"transformer_{name}.json")
        assert np.allclose(saved.mean_, tf.mean_, atol=1e-12) and np.allclose(saved.scale_, tf.scale_, atol=1e-12)


def test_experiment_loader_returns_no_labels_and_right_shapes():
    data = load_experiment_inputs(ROOT, CFG, "e1")
    assert set(data) == {"X_train", "X_val", "X_test", "test_index", "transformer", "feature_names"}
    assert data["X_train"].shape == (199_021, 29) and data["X_val"].shape == (42_647, 29) and data["X_test"].shape == (42_720, 29)
    assert data["X_train"].dtype == np.float32


def test_inverse_transform_handles_synthetic_like_vectors():
    tf = FeatureTransformer.load(processed_dir(ROOT, CFG) / "transformer_e1.json")
    z = np.random.default_rng(0).normal(size=(1000, 29)) * 3
    out = tf.inverse_transform(z)
    assert list(out.columns) == [*PCA_FEATURES, "Amount"]
    assert (out["Amount"] >= 0).all() and np.allclose(out["Amount"], out["Amount"].round(2))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
