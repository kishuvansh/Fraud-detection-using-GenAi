"""Training figures, test-set anomaly detection for E1 and E2 (all seeds), summaries and comparison.

Order per run: scores -> threshold (saved at training time) -> predictions written -> labels -> metrics.

Usage (from the project root):  python scripts/evaluate_detection.py
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
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.data.prepare import load_experiment_inputs  # noqa: E402
from src.detection.evaluate import evaluate_predictions, score_and_predict  # noqa: E402
from src.evaluation import model_plots as mp  # noqa: E402
from src.evaluation.detection_metrics import paired_bootstrap_ap_difference  # noqa: E402
from src.evaluation.plots import FRAUD_COLOR, NORMAL_COLOR, apply_style  # noqa: E402
from src.training.run import load_experiment_config, load_trained_model, read_run, run_dir  # noqa: E402
from src.utils.io import load_config, save_json  # noqa: E402

EXPERIMENTS = {"e1": "e1_primary", "e2": "e2_unsupervised"}
LABELS = {"e1": "E1 (normal-only)", "e2": "E2 (unsupervised)"}


def training_figures(exp: str, exp_cfg: dict, fig_dir: Path) -> None:
    histories, best = {}, {}
    for seed in exp_cfg["seeds"]:
        rd = run_dir(ROOT, exp, seed)
        histories[seed] = pd.read_csv(rd / "history.csv")
        best[seed] = read_run(rd)["meta"]["best_epoch"]
    hs = exp_cfg["headline_seed"]
    warm = exp_cfg["training"]["kl_warmup_epochs"]
    mp.plot_training_curves(histories[hs], best[hs], warm, f"{LABELS[exp]} training curves (seed {hs})", fig_dir / f"{exp}_training_curves.png")
    mp.plot_seed_curves(histories, best, f"{LABELS[exp]}: validation loss across seeds", fig_dir / f"{exp}_validation_loss_seeds.png")
    run = read_run(run_dir(ROOT, exp, hs))
    mp.plot_validation_scores(run["val_scores"], run["threshold"]["threshold"], run["threshold"]["percentile"],
                              f"{LABELS[exp]}: reconstruction error on {run['threshold']['validation_rows_description']} (seed {hs})",
                              fig_dir / f"{exp}_validation_score_distribution.png")
    mp.plot_kl_per_dim(np.array(run["meta"]["kl_per_latent_dim"]), f"{LABELS[exp]}: latent dimension usage (seed {hs})",
                       fig_dir / f"{exp}_latent_kl_per_dim.png")
    plt.close("all")


def main() -> None:
    data_cfg = load_config(ROOT)
    apply_style()
    fig_dir = ROOT / "outputs" / "figures" / "detection"
    train_fig_dir = ROOT / "outputs" / "figures" / "training"
    metrics_dir = ROOT / "outputs" / "metrics"
    results, rows = {}, []

    for exp, cfg_name in EXPERIMENTS.items():
        exp_cfg = load_experiment_config(ROOT, cfg_name)
        data = load_experiment_inputs(ROOT, data_cfg, exp)
        training_figures(exp, exp_cfg, train_fig_dir)
        # Steps 1-3 for every seed BEFORE any label is read
        for seed in exp_cfg["seeds"]:
            score_and_predict(ROOT, data_cfg, exp_cfg, seed, data)
        # Steps 4-5
        for seed in exp_cfg["seeds"]:
            res = evaluate_predictions(ROOT, data_cfg, exp_cfg, seed)
            results[(exp, seed)] = res
            m, meta = res["metrics"], read_run(run_dir(ROOT, exp, seed))["meta"]
            p = m["primary"]
            rows.append({"experiment": exp, "seed": seed, "average_precision": p["average_precision"], "roc_auc": p["roc_auc"],
                         "precision": p["precision"], "recall": p["recall"], "f1": p["f1"], "fpr": p["fpr"], "tpr": p["tpr"],
                         "tp": p["tp"], "fp": p["fp"], "fn": p["fn"], "tn": p["tn"], "threshold": p["threshold"],
                         "ap_neg_elbo": m["secondary_scores"]["neg_elbo"]["average_precision"],
                         "ap_iwae_nll": m["secondary_scores"]["iwae_nll"]["average_precision"],
                         "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"], "runtime_s": meta["runtime_seconds"]})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(metrics_dir / "detection_per_seed.csv", index=False)
    metric_cols = ["average_precision", "roc_auc", "precision", "recall", "f1", "fpr", "tpr", "ap_neg_elbo", "ap_iwae_nll"]
    summary = per_seed.groupby("experiment")[metric_cols].agg(["mean", "std"])
    summary.to_csv(metrics_dir / "detection_summary_mean_std.csv")

    # Headline figures (pre-registered seed 0 of each experiment)
    e1_cfg = load_experiment_config(ROOT, "e1_primary")
    hs = e1_cfg["headline_seed"]
    r1, r2 = results[("e1", hs)], results[("e2", hs)]
    y = r1["y_test"]
    s1, s2 = r1["scores"]["test_recon_mse"], r2["scores"]["test_recon_mse"]
    m1, m2 = r1["metrics"]["primary"], r2["metrics"]["primary"]
    prev = float(y.mean())

    mp.plot_pr_curve({f"E1 reconstruction error (AP {m1['average_precision']:.3f})": (y, s1, NORMAL_COLOR)}, prev,
                     {f"τ (val p99): P {m1['precision']:.2f}, R {m1['recall']:.2f}": (m1["recall"], m1["precision"])},
                     f"Precision–recall curve, test set (E1, seed {hs})", fig_dir / "e1_precision_recall_curve.png")
    mp.plot_roc_curve({f"E1 (ROC-AUC {m1['roc_auc']:.3f})": (y, s1, NORMAL_COLOR)},
                      {f"τ: FPR {m1['fpr']:.3f}, TPR {m1['tpr']:.2f}": (m1["fpr"], m1["tpr"])},
                      f"ROC curve, test set (E1, seed {hs})", fig_dir / "e1_roc_curve.png")
    mp.plot_confusion_matrix(m1, f"Confusion matrix, test set (E1, seed {hs})", fig_dir / "e1_confusion_matrix.png")
    mp.plot_score_by_class(y, s1, m1["threshold"], f"Reconstruction error by class, test set (E1, seed {hs})", fig_dir / "e1_score_by_class.png")
    mp.plot_score_distribution(s1, r1["scores"]["val_recon_mse"], m1["threshold"], f"Anomaly-score distribution (E1, seed {hs})",
                               fig_dir / "e1_score_distribution.png")
    mp.plot_threshold_curves(y, s1, r1["sensitivity"], f"Threshold sensitivity on the test set (E1, seed {hs}); thresholds from validation only",
                             fig_dir / "e1_threshold_sensitivity.png")
    mp.plot_pr_curve({f"E1 normal-only (AP {m1['average_precision']:.3f})": (y, s1, NORMAL_COLOR),
                      f"E2 unsupervised (AP {m2['average_precision']:.3f})": (y, s2, mp.NEUTRAL)}, prev, {},
                     f"E1 vs E2 precision–recall curves (seed {hs})", fig_dir / "e1_vs_e2_precision_recall.png")
    ap_table = per_seed.pivot(index="seed", columns="experiment", values="average_precision").rename(columns=LABELS)
    mp.plot_seed_comparison(ap_table, "Average precision (test)", "E1 vs E2 across 5 seeds", fig_dir / "e1_vs_e2_ap_by_seed.png")
    mp.plot_score_by_class(y, s2, m2["threshold"], f"Reconstruction error by class, test set (E2, seed {hs})", fig_dir / "e2_score_by_class.png")

    # Latent space (coloured by class after evaluation)
    model = load_trained_model(run_dir(ROOT, "e1", hs), "cpu")
    X_test = load_experiment_inputs(ROOT, data_cfg, "e1")["X_test"]
    with torch.no_grad():
        mu = model.encode(torch.from_numpy(X_test))[0].numpy()
    mu2 = PCA(n_components=2, random_state=0).fit_transform(mu)
    mp.plot_latent_space(mu2, y, "PCA of latent mean μ, component", f"Latent space of test transactions (E1, seed {hs})",
                         fig_dir / "e1_latent_space.png")
    plt.close("all")

    comparison = {
        "per_seed_ap": ap_table.to_dict(),
        "mean_ap": {LABELS[e]: float(per_seed.loc[per_seed.experiment == e, "average_precision"].mean()) for e in LABELS},
        "seeds_where_e1_ap_higher": int((ap_table[LABELS["e1"]] > ap_table[LABELS["e2"]]).sum()),
        "paired_bootstrap_headline_seed": paired_bootstrap_ap_difference(y, s1, s2, e1_cfg["evaluation"]["bootstrap_resamples"],
                                                                         e1_cfg["evaluation"]["bootstrap_seed"]),
    }
    save_json(comparison, metrics_dir / "detection_e1_vs_e2.json")
    r1["sensitivity"].to_csv(metrics_dir / "e1_threshold_sensitivity_headline.csv", index=False)
    save_json(r1["metrics"], metrics_dir / "e1_headline_metrics.json")
    save_json(r2["metrics"], metrics_dir / "e2_headline_metrics.json")

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(comparison["paired_bootstrap_headline_seed"])


if __name__ == "__main__":
    main()
