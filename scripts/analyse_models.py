"""E1 vs E2 comparison and additional analysis (latent space, reconstruction behaviour, class imbalance).

Runs after scripts/evaluate_detection.py (test predictions are already saved, so reading labels here is evaluation).
Outputs: outputs/metrics/comparison_*.csv|json, outputs/figures/analysis/*.png

Usage (from the project root):  python scripts/analyse_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import use_project_matplotlib_cache  # noqa: E402

use_project_matplotlib_cache(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.data.prepare import load_experiment_inputs, load_test_labels  # noqa: E402
from src.evaluation import analysis as an  # noqa: E402
from src.evaluation import model_plots as mp  # noqa: E402
from src.evaluation.detection_metrics import recall_at_budget  # noqa: E402
from src.evaluation.plots import apply_style  # noqa: E402
from src.training.run import load_trained_model, read_run, run_dir  # noqa: E402
from src.utils.io import load_config, load_json, save_json  # noqa: E402

SEED = 0


def main() -> None:
    data_cfg = load_config(ROOT)
    apply_style()
    figs, metrics = ROOT / "outputs" / "figures" / "analysis", ROOT / "outputs" / "metrics"
    for exp in ("e1", "e2"):
        if not (run_dir(ROOT, exp, SEED) / "predictions_meta.json").exists():
            raise RuntimeError("Run scripts/evaluate_detection.py first (predictions must exist before labels are used)")
    y = load_test_labels(ROOT, data_cfg)
    per_seed = pd.read_csv(metrics / "detection_per_seed.csv")

    # ---- comparison table (mean ± std over 5 seeds) ----
    cols = {"average_precision": "PR-AUC (average precision)", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall",
            "f1": "F1-score", "fpr": "False-positive rate", "best_epoch": "Best epoch", "runtime_s": "Training time (s)"}
    rows = []
    for key, label in cols.items():
        a, b = per_seed.loc[per_seed.experiment == "e1", key], per_seed.loc[per_seed.experiment == "e2", key]
        rows.append({"metric": label, "E1 normal-only (mean)": a.mean(), "E1 std": a.std(), "E2 unsupervised (mean)": b.mean(),
                     "E2 std": b.std(), "difference E1 − E2": a.mean() - b.mean(),
                     "seeds where E1 higher": int((a.to_numpy() > b.to_numpy()).sum())})
    table = pd.DataFrame(rows).set_index("metric")
    table.to_csv(metrics / "comparison_table.csv")

    # ---- per-model artefacts for the headline seed ----
    models, inputs, scores, preds, hist, best, tau = {}, {}, {}, {}, {}, {}, {}
    for exp in ("e1", "e2"):
        rd = run_dir(ROOT, exp, SEED)
        models[exp] = load_trained_model(rd, "cpu")
        inputs[exp] = load_experiment_inputs(ROOT, data_cfg, exp)["X_test"]
        scores[exp] = np.load(rd / "scores.npz")["test_recon_mse"]
        preds[exp] = pd.read_csv(rd / "test_predictions.csv")["predicted_anomaly"].to_numpy()
        hist[exp] = pd.read_csv(rd / "history.csv")
        run = read_run(rd)
        best[exp], tau[exp] = run["meta"]["best_epoch"], run["threshold"]["threshold"]

    dist_table = an.score_distribution_table({"E1 normal-only": scores["e1"], "E2 unsupervised": scores["e2"]}, y)
    dist_table.to_csv(metrics / "comparison_score_distributions.csv")
    overlap = an.detection_overlap(y, preds["e1"], preds["e2"])
    feature_names = [f"V{i}" for i in range(1, 29)] + ["log1p_Amount"]
    recon = {exp: an.reconstructions(models[exp], inputs[exp]) for exp in models}
    err_tables = {exp: an.per_feature_error_table(inputs[exp], recon[exp], y, feature_names) for exp in models}
    err_tables["e1"].to_csv(metrics / "analysis_per_feature_error_e1.csv")
    err_tables["e2"].to_csv(metrics / "analysis_per_feature_error_e2.csv")
    mu = {exp: an.latent_means(models[exp], inputs[exp])[0] for exp in models}
    pcas = {exp: PCA(n_components=2, random_state=0).fit(mu[exp]) for exp in models}
    mu2d = {exp: pcas[exp].transform(mu[exp]) for exp in models}
    latent = {exp: {**an.latent_summary(mu[exp], y), "pca_explained_variance": pcas[exp].explained_variance_ratio_.tolist(),
                    "active_dims": read_run(run_dir(ROOT, exp, SEED))["meta"]["active_latent_dims"]} for exp in models}

    e1m = load_json(metrics / "e1_headline_metrics.json")["primary"]
    prev = float(y.mean())
    grid = np.logspace(-4, np.log10(0.5), 200)
    prec_curve = an.precision_vs_prevalence(e1m["tpr"], e1m["fpr"], grid)

    summary = {
        "seed": SEED,
        "training": {exp: {"best_epoch": best[exp], "train_recon_at_best": float(hist[exp].loc[hist[exp].epoch == best[exp], "train_recon"].iloc[0]),
                           "train_kl_at_best": float(hist[exp].loc[hist[exp].epoch == best[exp], "train_kl"].iloc[0])} for exp in models},
        "score_distributions": dist_table.reset_index().to_dict(orient="records"),
        "detection_overlap_seed0": overlap,
        "recall_at_budget": {exp: recall_at_budget(y, scores[exp], [100, 500, 1000]) for exp in models},
        "latent": latent,
        "top_fraud_error_features_e1": err_tables["e1"].head(6)["fraud_mean_sq_error"].round(3).to_dict(),
        "top_fraud_error_features_e2": err_tables["e2"].head(6)["fraud_mean_sq_error"].round(3).to_dict(),
        "fraud_error_share_top5_features_e1": float(err_tables["e1"].head(5)["fraud_share_of_total_error"].sum()),
        "imbalance": {"test_prevalence": prev, "tpr": e1m["tpr"], "fpr": e1m["fpr"],
                      "expected_precision": {f"{p:g}": float(an.precision_vs_prevalence(e1m["tpr"], e1m["fpr"], np.array([p]))[0])
                                             for p in (prev, 0.01, 0.05, 0.1, 0.5)},
                      "false_alarms_per_fraud_caught": e1m["fp"] / e1m["tp"]},
    }
    save_json(summary, metrics / "comparison_analysis.json")

    # ---- figures ----
    mp.plot_training_comparison(hist["e1"], hist["e2"], best["e1"], best["e2"], "Training behaviour: E1 vs E2 (seed 0)", figs / "training_comparison.png")
    mp.plot_score_distributions_pair(y, scores["e1"], scores["e2"], tau["e1"], tau["e2"], "Test reconstruction-error distributions: E1 vs E2 (seed 0)",
                                     figs / "score_distributions_e1_vs_e2.png")
    mp.plot_metric_comparison(per_seed, "Test metrics: E1 normal-only vs E2 unsupervised (5 seeds)", figs / "metric_comparison.png")
    mp.plot_latent_pair(mu2d["e1"], mu2d["e2"], y, "Latent space of test transactions (PCA of μ; coloured after training)", figs / "latent_space_e1_vs_e2.png")
    mp.plot_latent_dims(mu["e1"], y, "E1 latent dimensions: posterior mean μ for normal vs fraud test transactions", figs / "latent_dims_e1.png")
    mp.plot_per_feature_errors(err_tables["e1"], err_tables["e2"], "Per-feature reconstruction error on test transactions (seed 0)", figs / "per_feature_errors.png")
    top = [feature_names.index(f) for f in err_tables["e1"].index[:3]]
    mp.plot_reconstruction_scatter(inputs["e1"], recon["e1"], y, top, feature_names, "E1 reconstruction behaviour: actual vs reconstructed (dotted line = perfect)",
                                   figs / "reconstruction_scatter_e1.png")
    mp.plot_reconstruction_scatter(inputs["e2"], recon["e2"], y, [feature_names.index(f) for f in err_tables["e1"].index[:3]], feature_names,
                                   "E2 reconstruction behaviour on the same features", figs / "reconstruction_scatter_e2.png")
    mp.plot_precision_vs_prevalence(grid, prec_curve, prev, e1m["tpr"], e1m["fpr"], "Effect of class imbalance on precision", figs / "precision_vs_prevalence.png")
    plt.close("all")

    pd.set_option("display.width", 220)
    print(table.round(4).to_string())
    print(dist_table.round(3).to_string())
    print("overlap:", overlap)
    print("latent:", latent)
    print("top fraud-error features E1:", summary["top_fraud_error_features_e1"], "share top5:", round(summary["fraud_error_share_top5_features_e1"], 3))
    print("imbalance:", summary["imbalance"])
    print("budgets:", summary["recall_at_budget"])


if __name__ == "__main__":
    main()
