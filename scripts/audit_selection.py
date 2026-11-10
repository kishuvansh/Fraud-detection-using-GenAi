"""Audit before model selection: split usage history and a SHA-256 freeze manifest of the baseline artefacts.

Usage:  python scripts/audit_selection.py            # writes the audit + manifest
        python scripts/audit_selection.py --verify   # re-checks that frozen baseline files are unchanged
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import TARGET_COL, load_raw  # noqa: E402
from src.data.split import load_splits  # noqa: E402
from src.utils.io import load_config, save_json, sha256_file  # noqa: E402

OUT = ROOT / "outputs" / "model_selection"
FROZEN_GLOBS = ["outputs/runs/e1/**/*", "outputs/runs/e2/**/*", "outputs/metrics/*", "data/processed/*", "data/synthetic/*"]


def manifest() -> dict:
    files = sorted({p for g in FROZEN_GLOBS for p in ROOT.glob(g) if p.is_file()})
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        frozen = json.loads((OUT / "baseline_manifest.json").read_text())["files"]
        now = manifest()
        changed = [f for f, h in frozen.items() if now.get(f) != h]
        print(f"{len(frozen)} frozen files checked; changed or missing: {changed or 'none'}")
        sys.exit(1 if changed else 0)

    cfg = load_config(ROOT)
    df = load_raw(ROOT, cfg)
    y = df[TARGET_COL].to_numpy()
    splits, meta = load_splits(ROOT / "data" / "processed" / "splits.json")
    history = {}
    for exp in ("e1", "e2"):
        pm = json.loads((ROOT / "outputs" / "runs" / exp / "seed_0" / "predictions_meta.json").read_text())
        history[exp] = {"validation_rows_scored": int(len(np.load(ROOT / "outputs" / "runs" / exp / "seed_0" / "val_scores.npy"))),
                        "test_predictions_written_utc": pm["written_at_utc"]}
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {s: {"rows": int(len(v)), "normal": int((y[v] == 0).sum()), "fraud": int(y[v].sum())} for s, v in splits.items()},
        "split_fingerprints": meta["index_sha256"],
        "validation_fraud_usage_so_far": {
            "e1": "never scored: E1 calibration and early stopping used the 42,647 normal validation rows only",
            "e2": "scored without labels (all 42,721 validation rows) to set E2's percentile threshold; labels never read",
            "metrics_using_validation_labels": "none exist before this audit",
        },
        "test_set_history": {
            "test_labels_read_for": ["E1 and E2 final evaluation (scripts/evaluate_detection.py)",
                                     "post-hoc analysis (scripts/analyse_models.py)",
                                     "selecting normal test rows as the real reference for generation quality"],
            "runs": history,
        },
        "disclosure": ("Test results of the baseline are already known. Two of the candidate ideas were suggested by earlier analyses that "
                       "used test labels: the ELBO score (test AP 0.450 vs 0.432 for reconstruction error) and feature-wise normalisation "
                       "(V27 dominates the fraud reconstruction error on test). Candidates suggested by test observations may look "
                       "optimistic on the test set. Mitigation: the candidate list is fixed in configs/model_selection.yaml before any "
                       "validation result is computed, selection uses validation data only, and the test set is evaluated once afterwards."),
    }
    save_json(audit, OUT / "audit.json")
    m = manifest()
    save_json({"created_utc": audit["created_utc"], "n_files": len(m), "files": m}, OUT / "baseline_manifest.json")
    print(json.dumps({k: audit[k] for k in ("split", "validation_fraud_usage_so_far")}, indent=2))
    print(f"baseline manifest: {len(m)} files frozen")


if __name__ == "__main__":
    main()
