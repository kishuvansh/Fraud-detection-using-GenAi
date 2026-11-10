"""Test-set scoring and evaluation, with the order of operations enforced (PROJECT_PLAN.md §3.5):

    1. compute test scores   2. apply the saved validation threshold   3. write predictions to disk
    4. only then load test labels   5. compute metrics

Step 1-3 live in `score_and_predict` (no labels anywhere); steps 4-5 live in `evaluate_predictions`,
which refuses to run unless the prediction file exists and is unchanged since it was written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare import load_experiment_inputs, load_test_labels
from src.detection.scores import all_scores
from src.detection.threshold import percentile_threshold
from src.evaluation.detection_metrics import (bootstrap_ci, ranking_metrics, recall_at_budget, sensitivity_table,
                                              threshold_metrics)
from src.training.run import load_trained_model, read_run, run_dir
from src.utils.io import load_json, save_json, sha256_file
from src.utils.runtime import resolve_device

SCORE_NAMES = ("recon_mse", "neg_elbo", "iwae_nll")


def score_and_predict(root: Path, data_cfg: dict, exp_cfg: dict, seed: int, data: dict | None = None) -> dict:
    """Steps 1-3. Uses only features and the threshold saved at training time."""
    experiment = exp_cfg["experiment"]
    out = run_dir(root, experiment, seed)
    device = resolve_device(exp_cfg.get("device", "cpu"))
    data = data or load_experiment_inputs(root, data_cfg, experiment)
    model = load_trained_model(out, device)
    run = read_run(out)
    tau = run["threshold"]["threshold"]

    test_scores = all_scores(model, data["X_test"], device, exp_cfg["scores"])
    val_scores = all_scores(model, data["X_val"], device, exp_cfg["scores"])
    if not np.allclose(val_scores["recon_mse"], run["val_scores"], rtol=1e-6, atol=1e-9):
        raise RuntimeError("Recomputed validation scores differ from those used for threshold calibration")
    np.savez(out / "scores.npz", **{f"test_{k}": v for k, v in test_scores.items()},
             **{f"val_{k}": v for k, v in val_scores.items()})

    predictions = pd.DataFrame({
        "dataset_row": data["test_index"],
        "recon_mse": test_scores["recon_mse"],
        "predicted_anomaly": (test_scores["recon_mse"] > tau).astype(int),
    })
    pred_path = out / "test_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    save_json({"threshold": tau, "score": "recon_mse", "rows": len(predictions),
               "flagged": int(predictions["predicted_anomaly"].sum()),
               "sha256": sha256_file(pred_path), "written_at_utc": datetime.now(timezone.utc).isoformat(),
               "labels_used": False}, out / "predictions_meta.json")
    return {"predictions": predictions, "test_scores": test_scores, "val_scores": val_scores, "threshold": tau}


def evaluate_predictions(root: Path, data_cfg: dict, exp_cfg: dict, seed: int) -> dict:
    """Steps 4-5. Labels are loaded only after confirming the predictions were saved beforehand."""
    out = run_dir(root, exp_cfg["experiment"], seed)
    pred_path, meta_path = out / "test_predictions.csv", out / "predictions_meta.json"
    if not pred_path.exists() or not meta_path.exists():
        raise RuntimeError("Test predictions must be written before test labels are loaded")
    pred_meta = load_json(meta_path)
    if sha256_file(pred_path) != pred_meta["sha256"]:
        raise RuntimeError("Prediction file changed after it was written")

    y = load_test_labels(root, data_cfg)          # <- first and only access to test labels
    predictions = pd.read_csv(pred_path)
    scores = np.load(out / "scores.npz")
    tau = pred_meta["threshold"]
    ev = exp_cfg["evaluation"]
    primary_q = exp_cfg["threshold"]["percentile"]

    primary = {**threshold_metrics(y, predictions["recon_mse"].to_numpy(), tau), **ranking_metrics(y, predictions["recon_mse"].to_numpy())}
    assert primary["flagged"] == int(predictions["predicted_anomaly"].sum())
    metrics = {
        "experiment": exp_cfg["experiment"], "seed": seed, "test_rows": len(y), "test_frauds": int(y.sum()),
        "primary_score": "recon_mse", "threshold_source": "99th percentile of validation scores (no labels)",
        "primary": primary,
        "bootstrap_95ci": bootstrap_ci(y, predictions["recon_mse"].to_numpy(), tau, ev["bootstrap_resamples"], ev["bootstrap_seed"]),
        "recall_at_budget": recall_at_budget(y, predictions["recon_mse"].to_numpy(), ev["alert_budgets"]),
        "secondary_scores": {},
    }
    for name in SCORE_NAMES[1:]:
        s_test, s_val = scores[f"test_{name}"], scores[f"val_{name}"]
        tau_s = percentile_threshold(s_val, primary_q)
        metrics["secondary_scores"][name] = {**ranking_metrics(y, s_test), **threshold_metrics(y, s_test, tau_s)}
    sens = sensitivity_table(y, scores["test_recon_mse"], scores["val_recon_mse"], ev["sensitivity_percentiles"], primary_q)
    sens.to_csv(out / "threshold_sensitivity.csv", index=False)
    save_json(metrics, out / "metrics.json")
    return {"metrics": metrics, "sensitivity": sens, "y_test": y, "scores": {k: scores[k] for k in scores.files}}
