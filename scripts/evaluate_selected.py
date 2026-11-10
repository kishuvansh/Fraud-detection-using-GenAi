"""Held-out test evaluation AFTER the selection lock: frozen baseline vs selected candidate, then the ablation study.

Refuses to run unless outputs/model_selection/selection_lock.json exists, the protocol hash is unchanged and the
frozen baseline files are unchanged. Nothing here feeds back into selection or thresholds.

Usage:  python scripts/evaluate_selected.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402

from src.data.prepare import load_test_labels  # noqa: E402
from src.evaluation import model_plots as mp  # noqa: E402
from src.evaluation.detection_metrics import bootstrap_ci, paired_bootstrap_ap_difference, threshold_metrics  # noqa: E402
from src.evaluation.plots import NORMAL_COLOR, TEXT, apply_style  # noqa: E402
from src.experiments.variants import (VariantData, is_variational, load_protocol, load_variant_model, score_set,  # noqa: E402
                                      train_feature_mse, variant_config, variant_dir)
from src.generation.sample import generate_model_space  # noqa: E402
from src.utils.io import load_config, load_json, save_json, sha256_file  # noqa: E402

OUT = ROOT / "outputs" / "model_selection"
FIG = ROOT / "outputs" / "figures" / "model_selection"


def preconditions() -> dict:
    lock_path = OUT / "selection_lock.json"
    if not lock_path.exists():
        raise RuntimeError("No selection lock: run scripts/select_model.py first. Test labels stay closed.")
    lock = load_json(lock_path)
    reg = load_json(OUT / "protocol_registration.json")
    if sha256_file(ROOT / reg["file"]) != lock["protocol_sha256"]:
        raise RuntimeError("Protocol changed after selection")
    if sha256_file(OUT / "selection_table.csv") != lock["selection_table_sha256"]:
        raise RuntimeError("Selection table changed after the lock")
    frozen = load_json(OUT / "baseline_manifest.json")["files"]
    changed = [f for f, h in frozen.items() if not (ROOT / f).exists() or sha256_file(ROOT / f) != h]
    if changed:
        raise RuntimeError(f"Frozen baseline files changed: {changed[:5]}")
    return lock


def generation_quality(model, tf, X_val_normal_model_space: np.ndarray, seed: int = 0, n: int = 20000) -> dict:
    """Decoder-mean samples from z ~ N(0, I) compared with real normal validation rows."""
    Z = generate_model_space(model, n, seed)
    synth = tf.inverse_transform(Z)
    real_z = X_val_normal_model_space.astype(np.float64)
    synth_z = tf.transform(synth)
    ks = np.mean([ks_2samp(real_z[:, j], synth_z[:, j]).statistic for j in range(real_z.shape[1])])
    real = tf.inverse_transform(real_z)
    std_ratio = float((synth.iloc[:, :28].std() / real.iloc[:, :28].std()).mean())
    return {"gen_mean_ks": float(ks), "gen_std_ratio_V": std_ratio}


def main() -> None:
    lock = preconditions()
    data_cfg, protocol = load_config(ROOT), load_protocol(ROOT)
    seeds = protocol["seeds"]
    sel = lock["selected"]
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    events = [{"time_utc": lock["locked_utc"], "event": "selection locked (no test labels accessed)"}]
    data = {"log1p": VariantData(ROOT, data_cfg, "log1p")}

    # ---- test scores for the selected candidate (computed before labels are read) ----
    sel_scores = []
    for seed in seeds:
        model = load_variant_model(ROOT, sel["architecture"], seed)
        fm = train_feature_mse(model, data["log1p"].X_train)
        sel_scores.append(score_set(model, data["log1p"].X_test, fm, is_variational(ROOT, sel["architecture"], seed))[sel["score"]])
    if sel["aggregation"] == "ensemble5":
        final = {"ensemble5": np.mean(sel_scores, axis=0)}
    else:
        final = {f"seed_{s}": sc for s, sc in zip(seeds, sel_scores)}
    preds = {k: (v > sel["thresholds"][k]).astype(int) for k, v in final.items()}
    pd.DataFrame({k: v for k, v in final.items()} | {f"pred_{k}": p for k, p in preds.items()}).to_csv(OUT / "selected_test_predictions.csv", index=False)
    events.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": "selected-model test predictions saved"})

    y = load_test_labels(ROOT, data_cfg)
    events.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": "test labels read for final evaluation"})

    # ---- frozen baseline (E1, recon_mse, single models) ----
    base_scores = {s: np.load(variant_dir(ROOT, "baseline", s) / "scores.npz")["test_recon_mse"] for s in seeds}
    base_tau = {s: load_json(variant_dir(ROOT, "baseline", s) / "threshold.json")["threshold"] for s in seeds}
    base_rows = [{"seed": s, **threshold_metrics(y, base_scores[s], base_tau[s]), "average_precision": average_precision_score(y, base_scores[s]),
                  "roc_auc": roc_auc_score(y, base_scores[s])} for s in seeds]
    base_df = pd.DataFrame(base_rows)
    sel_rows = [{"key": k, **threshold_metrics(y, v, sel["thresholds"][k]), "average_precision": average_precision_score(y, v),
                 "roc_auc": roc_auc_score(y, v)} for k, v in final.items()]
    sel_df = pd.DataFrame(sel_rows)
    cols = ["average_precision", "roc_auc", "precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]
    sel_key = "ensemble5" if sel["aggregation"] == "ensemble5" else "seed_0"
    comparison = {
        "baseline": {"description": "frozen E1: latent 8, beta 1, recon_mse, single model; 99th pct of normal-validation scores",
                     "mean": base_df[cols].mean().to_dict(), "std": base_df[cols].std().to_dict(),
                     "seed0": base_rows[0], "seed0_95ci": bootstrap_ci(y, base_scores[0], base_tau[0], 1000, 7)},
        "selected": {"description": f"{sel['architecture']} / {sel['score']} / {sel['aggregation']}", "rows": sel_rows,
                     "mean": sel_df[cols].mean().to_dict(), "std": sel_df[cols].std().to_dict() if len(sel_df) > 1 else None,
                     "headline_95ci": bootstrap_ci(y, final[sel_key], sel["thresholds"][sel_key], 1000, 7)},
        "paired_bootstrap_ap_selected_minus_baseline_seed0": paired_bootstrap_ap_difference(y, final[sel_key], base_scores[0], 1000, 7),
    }

    # ---- ablation study (analysis only; selection already locked) ----
    abl_names = ["baseline", *protocol["ablation"]["configurations"].keys()]
    abl_rows = []
    for name in abl_names:
        transform = variant_config(ROOT, protocol, name)["preprocessing"]["amount_transform"]
        if transform not in data:
            data[transform] = VariantData(ROOT, data_cfg, transform)
        d = data[transform]
        for seed in seeds:
            model = load_variant_model(ROOT, name, seed)
            variational = is_variational(ROOT, name, seed)
            fm = train_feature_mse(model, d.X_train)
            v = score_set(model, d.X_val_all, fm, False)["recon_mse"]
            t = score_set(model, d.X_test, fm, False)["recon_mse"]
            tau = float(np.percentile(v[d.val_normal_mask], 99.0))
            meta = load_json(variant_dir(ROOT, name, seed) / "train_meta.json")
            row = {"configuration": name, "seed": seed, "variational": variational, "val_ap": average_precision_score(d.y_val_all, v),
                   "test_ap": average_precision_score(y, t), "test_roc_auc": roc_auc_score(y, t),
                   **{f"test_{k}": val for k, val in threshold_metrics(y, t, tau).items() if k in ("precision", "recall", "f1", "fpr", "tp", "fp")},
                   "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"], "runtime_s": meta["runtime_seconds"],
                   "active_latent_dims": meta.get("active_latent_dims")}
            row.update(generation_quality(model, d.transformer, d.X_val_normal))
            abl_rows.append(row)
        print(f"ablation evaluated: {name}")
    e2 = pd.read_csv(ROOT / "outputs" / "metrics" / "detection_per_seed.csv").query("experiment == 'e2'")
    for _, r in e2.iterrows():
        abl_rows.append({"configuration": "all_training_data (E2)", "seed": int(r.seed), "variational": True, "val_ap": np.nan,
                         "test_ap": r.average_precision, "test_roc_auc": r.roc_auc, "test_precision": r.precision, "test_recall": r.recall,
                         "test_f1": r.f1, "test_fpr": r.fpr, "test_tp": r.tp, "test_fp": r.fp, "best_epoch": r.best_epoch, "runtime_s": r.runtime_s})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(OUT / "ablation_per_seed.csv", index=False)
    metric_cols = ["val_ap", "test_ap", "test_roc_auc", "test_precision", "test_recall", "test_f1", "gen_mean_ks", "gen_std_ratio_V", "active_latent_dims", "best_epoch"]
    abl_summary = abl.groupby("configuration", sort=False)[metric_cols].agg(["mean", "std"])
    abl_summary.to_csv(OUT / "ablation_summary.csv")

    events.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": "ablation evaluated (analysis only)"})
    save_json({"lock": lock, "comparison": comparison, "events": events}, OUT / "final_test_results.json")

    # ---- figures ----
    sel_label = f"Selected: {sel['architecture']} / {sel['score']} / {sel['aggregation']}"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    for s_, color, label in ((base_scores[0], mp.NEUTRAL, f"Baseline seed 0 (AP {average_precision_score(y, base_scores[0]):.3f})"),
                             (final[sel_key], NORMAL_COLOR, f"{sel_label} (AP {average_precision_score(y, final[sel_key]):.3f})")):
        p, r, _ = precision_recall_curve(y, s_)
        axes[0].plot(r, p, color=color, linewidth=2, label=label)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].legend(fontsize=7); axes[0].set_title("Test precision–recall curves")
    for ax, (m, title) in zip(axes[1:], ((base_rows[0], "Baseline (seed 0)"), (sel_rows[0], sel_label))):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]); share = cm / cm.sum(1, keepdims=True)
        ax.imshow(share, cmap=mp.SEQ_BLUE, vmin=0, vmax=1)
        labels = [["True negative", "False positive"], ["False negative", "True positive"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({100 * share[i, j]:.1f}% of row)", ha="center", va="center", fontsize=9,
                        color="#fcfcfb" if share[i, j] > 0.5 else TEXT)
        ax.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"]); ax.set_yticks([0, 1], ["Actual normal", "Actual fraud"]); ax.grid(visible=False)
        ax.set_title(f"{title}\nτ = {m['threshold']:.3f} · P {m['precision']:.3f} · R {m['recall']:.3f}", fontsize=9)
    fig.suptitle("Held-out test set: frozen baseline vs validation-selected model")
    fig.savefig(FIG / "baseline_vs_selected_test.png", dpi=150, bbox_inches="tight")

    table = pd.read_csv(OUT / "selection_table.csv")
    table["label"] = table.architecture + " · " + table.score + " · " + table.aggregation
    t2 = table.sort_values("val_ap")
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    colors = [NORMAL_COLOR if (r.architecture == sel["architecture"] and r.score == sel["score"] and r.aggregation == sel["aggregation"])
              else (TEXT if (r.architecture == "baseline" and r.score == "recon_mse" and r.aggregation == "single") else mp.NEUTRAL) for r in t2.itertuples()]
    ax.barh(t2.label, t2.val_ap, color=colors, height=0.7)
    ax.set_xlabel("Validation average precision (42,647 normal + 74 fraud)"); ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", visible=False)
    ax.set_title("Blue = selected · black = frozen baseline configuration", fontsize=9)
    fig.suptitle("Model selection on validation data (30 pre-registered candidates)")
    fig.savefig(FIG / "validation_selection.png", dpi=150, bbox_inches="tight")

    order = ["baseline", "autoencoder", "no_kl_warmup", "raw_amount", "latent2", "latent4", "latent16", "beta0.5", "beta2", "all_training_data (E2)"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout="constrained")
    for ax, metric, title in ((axes[0], "test_ap", "Test PR-AUC (recon score; bars = mean, dots = seeds)"), (axes[1], "gen_mean_ks", "Generation: mean KS vs real normal (lower = better)")):
        sub = abl[abl.configuration.isin(order)]
        means = sub.groupby("configuration")[metric].mean().reindex(order)
        x = np.arange(len(order))
        ax.bar(x, means.values, color=[TEXT if o == "baseline" else NORMAL_COLOR for o in order], alpha=0.45, width=0.65)
        for i, o in enumerate(order):
            vals = sub.loc[sub.configuration == o, metric].dropna()
            ax.scatter(np.full(len(vals), i), vals, color=TEXT if o == "baseline" else NORMAL_COLOR, s=12, zorder=3)
            if not np.isnan(means.iloc[i]):
                ax.annotate(f"{means.iloc[i]:.3f}", (i, vals.max()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(x, [o.replace(" (E2)", "\n(E2)") for o in order], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(axis="x", visible=False)
    fig.suptitle("Ablation study: one component changed at a time (E1 protocol unless stated)")
    fig.savefig(FIG / "ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    pd.set_option("display.width", 240)
    print(json.dumps({k: comparison[k]["mean"] for k in ("baseline", "selected")}, indent=1, default=float))
    print("paired:", comparison["paired_bootstrap_ap_selected_minus_baseline_seed0"])
    print(abl_summary.round(4).to_string())


if __name__ == "__main__":
    main()
