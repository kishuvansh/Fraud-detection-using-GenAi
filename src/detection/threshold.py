"""Label-free threshold calibration from validation anomaly scores."""

from __future__ import annotations

import numpy as np


def percentile_threshold(val_scores: np.ndarray, percentile: float) -> float:
    """tau = the given percentile of validation scores (linear interpolation)."""
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")
    return float(np.percentile(val_scores, percentile))


def calibration_record(val_scores: np.ndarray, percentile: float, experiment: str, score_name: str) -> dict:
    tau = percentile_threshold(val_scores, percentile)
    return {
        "method": "validation_percentile",
        "score": score_name,
        "percentile": percentile,
        "threshold": tau,
        "validation_rows": int(len(val_scores)),
        "validation_rows_description": ("normal validation rows only (E1)" if experiment == "e1"
                                        else "all validation rows, labels ignored (E2)"),
        "validation_flag_rate": float(np.mean(val_scores > tau)),
        "validation_score_summary": {q: float(np.percentile(val_scores, q)) for q in (50, 90, 95, 99, 99.5, 99.9)},
        "fraud_labels_used": False,
        "rationale": ("tau is the score below which the chosen share of validation transactions fall, so about "
                      f"{100 - percentile:g}% of normal-looking traffic is flagged. It uses no fraud labels and no test data."),
    }
