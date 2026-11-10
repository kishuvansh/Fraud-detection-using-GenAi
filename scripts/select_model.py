"""Validation-based model selection (pre-registered in configs/model_selection.yaml). Never touches test labels.

Steps: verify the protocol hash -> score all validation rows for every candidate -> rank by validation AP ->
apply the tie rule -> compute the selected candidate's label-free threshold -> write a lock file.

Usage:  python scripts/select_model.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.evaluation.detection_metrics import threshold_metrics  # noqa: E402
from src.experiments.variants import (VariantData, is_variational, load_protocol, load_variant_model, score_set,  # noqa: E402
                                      train_feature_mse, variant_dir)
from src.utils.io import load_config, save_json, sha256_file  # noqa: E402

OUT = ROOT / "outputs" / "model_selection"
SCORE_ORDER = ["recon_mse", "norm_recon_mse", "neg_elbo"]
AGG_ORDER = ["single", "ensemble5"]


def verify_protocol() -> str:
    reg = json.loads((OUT / "protocol_registration.json").read_text())
    sha = sha256_file(ROOT / reg["file"])
    if sha != reg["sha256"]:
        raise RuntimeError("configs/model_selection.yaml changed after registration; selection would be invalid")
    return sha


def bootstrap_ap(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = [average_precision_score(y[i], s[i]) for i in
             (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    protocol_sha = verify_protocol()
    if (OUT / "selection_lock.json").exists():
        raise RuntimeError("Selection already locked; it must not be re-run")
    data_cfg, protocol = load_config(ROOT), load_protocol(ROOT)
    seeds = protocol["seeds"]
    data = VariantData(ROOT, data_cfg, "log1p")
    y, normal = data.y_val_all, data.val_normal_mask
    archs = list(protocol["selection"]["architectures"])
    pct = 99.0

    scores: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    normalisers = {}
    for arch in archs:
        scores[arch] = {}
        for seed in seeds:
            model = load_variant_model(ROOT, arch, seed)
            fm = train_feature_mse(model, data.X_train)
            normalisers[f"{arch}/seed_{seed}"] = fm.tolist()
            scores[arch][seed] = score_set(model, data.X_val_all, fm, is_variational(ROOT, arch, seed))
            d = OUT / "val_scores" / arch
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"seed_{seed}.npz", **scores[arch][seed])
        print(f"scored validation rows for {arch}")
    save_json(normalisers, OUT / "norm_recon_feature_normalisers.json")

    rows = []
    for arch in archs:
        for score in SCORE_ORDER:
            per_seed = [scores[arch][s][score] for s in seeds]
            aps = [average_precision_score(y, s) for s in per_seed]
            singles = [threshold_metrics(y, s, float(np.percentile(s[normal], pct))) for s in per_seed]
            rows.append({"architecture": arch, "score": score, "aggregation": "single", "val_ap": float(np.mean(aps)), "val_ap_std": float(np.std(aps, ddof=1)),
                         "val_roc_auc": float(np.mean([roc_auc_score(y, s) for s in per_seed])),
                         "val_precision": float(np.mean([m["precision"] for m in singles])), "val_recall": float(np.mean([m["recall"] for m in singles])),
                         "val_f1": float(np.mean([m["f1"] for m in singles]))})
            ens = np.mean(per_seed, axis=0)
            m = threshold_metrics(y, ens, float(np.percentile(ens[normal], pct)))
            rows.append({"architecture": arch, "score": score, "aggregation": "ensemble5", "val_ap": float(average_precision_score(y, ens)), "val_ap_std": np.nan,
                         "val_roc_auc": float(roc_auc_score(y, ens)), "val_precision": m["precision"], "val_recall": m["recall"], "val_f1": m["f1"]})
    table = pd.DataFrame(rows)
    table["rank"] = table["val_ap"].rank(ascending=False, method="min").astype(int)
    table = table.sort_values("val_ap", ascending=False).reset_index(drop=True)

    # tie rule: simplest candidate within 0.002 of the best
    best = table["val_ap"].max()
    tied = table[table["val_ap"] >= best - 0.002].copy()
    tied["complexity"] = list(zip(tied["aggregation"].map(AGG_ORDER.index), tied["score"].map(SCORE_ORDER.index), tied["architecture"].map(archs.index)))
    chosen = tied.sort_values("complexity").iloc[0]
    sel = {"architecture": chosen["architecture"], "score": chosen["score"], "aggregation": chosen["aggregation"], "val_ap": float(chosen["val_ap"])}

    # label-free threshold(s) for the selected candidate: 99th percentile on NORMAL validation rows
    per_seed = [scores[sel["architecture"]][s][sel["score"]] for s in seeds]
    if sel["aggregation"] == "ensemble5":
        final_scores = np.mean(per_seed, axis=0)
        thresholds = {"ensemble5": float(np.percentile(final_scores[normal], pct))}
    else:
        final_scores = per_seed[0]
        thresholds = {f"seed_{s}": float(np.percentile(p[normal], pct)) for s, p in zip(seeds, per_seed)}
    base_row = table[(table.architecture == "baseline") & (table.score == "recon_mse") & (table.aggregation == "single")].iloc[0]
    sel["val_ap_95ci"] = bootstrap_ap(y, final_scores)
    sel["thresholds"] = thresholds
    sel["threshold_rule"] = "99th percentile of the selected score on the 42,647 normal validation rows (no fraud labels)"

    table.to_csv(OUT / "selection_table.csv", index=False)
    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": protocol_sha,
        "selection_table_sha256": sha256_file(OUT / "selection_table.csv"),
        "candidates_evaluated": int(len(table)),
        "validation_rows": {"normal": int(normal.sum()), "fraud": int((y == 1).sum())},
        "selected": sel,
        "baseline_validation_ap": {"mean": float(base_row["val_ap"]), "std": float(base_row["val_ap_std"])},
        "test_labels_accessed_before_lock": False,
    }
    save_json(lock, OUT / "selection_lock.json")
    pd.set_option("display.width", 220)
    print(table.drop(columns=["rank"]).round(4).to_string())
    print("\nSELECTED:", json.dumps(sel, indent=2))
    print("baseline (recon, single) validation AP:", lock["baseline_validation_ap"])


if __name__ == "__main__":
    main()
