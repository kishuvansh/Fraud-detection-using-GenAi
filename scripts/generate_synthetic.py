"""Generate synthetic feature vectors with the headline E1 VAE, export CSVs, and evaluate generation quality.

Two samplers are produced and evaluated in the same way (neither is selected using test data):
  * decoder_mean (PRIMARY, pre-registered): x = decoder(z), z ~ N(0, I)
  * decoder_mean_plus_noise (pre-listed variant): x = decoder(z) + eps, eps ~ N(0, diag(sigma^2)),
    where sigma is the decoder noise level estimated from TRAINING residuals.
Real reference = held-out normal TEST transactions (never used for training, early stopping or thresholds).

Usage (from the project root):
    python scripts/generate_synthetic.py                 # n = number of normal test rows
    python scripts/generate_synthetic.py --n 10000 --seed 1
"""

from __future__ import annotations

import argparse
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

from src.data.load import RAW_INPUT_COLUMNS, TARGET_COL, load_raw  # noqa: E402
from src.data.prepare import processed_dir  # noqa: E402
from src.data.preprocess import FeatureTransformer  # noqa: E402
from src.data.split import derived_partitions, load_splits  # noqa: E402
from src.evaluation import generative_metrics as gm  # noqa: E402
from src.evaluation import model_plots as mp  # noqa: E402
from src.evaluation.plots import apply_style  # noqa: E402
from src.generation.sample import export_synthetic, fit_decoder_noise_std, generate_synthetic  # noqa: E402
from src.training.run import load_experiment_config, load_trained_model, read_run, run_dir  # noqa: E402
from src.utils.io import load_config, save_json  # noqa: E402

VARIANT_TITLES = {"decoder_mean": "decoder mean (primary)", "decoder_mean_plus_noise": "decoder mean + fitted noise (variant)"}


def evaluate_variant(name: str, synth: pd.DataFrame, real: pd.DataFrame, real_z: np.ndarray, train_z: np.ndarray,
                     tf: FeatureTransformer, gen_cfg: dict, seed: int, metrics_dir: Path, fig_dir: Path) -> dict:
    synth_z = tf.transform(synth)  # evaluate exactly what is exported (Amount clipped and rounded)
    moments = gm.moment_table(real, synth)
    distances = gm.distance_table(real_z, synth_z, tf.feature_names)
    quantiles = gm.quantile_table(real, synth)
    corr = gm.correlation_comparison(real, synth)
    proj = gm.pca_projection(real_z, synth_z)
    c2st = gm.classifier_two_sample_test(real_z, synth_z, gen_cfg["c2st_samples"], seed)
    memo = gm.memorisation_check(train_z, real_z, synth_z, gen_cfg["dcr_samples"], seed)

    moments.to_csv(metrics_dir / f"generation_{name}_moments.csv")
    distances.to_csv(metrics_dir / f"generation_{name}_distances.csv")
    quantiles.to_csv(metrics_dir / f"generation_{name}_quantiles.csv")
    title = VARIANT_TITLES[name]
    mp.plot_real_vs_synthetic_grid(real, synth, f"Real normal (held-out test) vs synthetic [{title}]: feature distributions (density; 0.5–99.5th pct view)",
                                   fig_dir / f"{name}_distributions.png")
    mp.plot_moment_comparison(moments, f"Mean and standard deviation: real normal vs synthetic [{title}]", fig_dir / f"{name}_moments.png")
    mp.plot_distance_bars(distances, f"Per-feature distribution distance, real normal vs synthetic [{title}]", fig_dir / f"{name}_distances.png")
    mp.plot_correlation_comparison(corr, list(real.columns), f"Correlation structure: real normal vs synthetic [{title}]", fig_dir / f"{name}_correlations.png")
    mp.plot_pca_overlay(proj, f"PCA projection fitted on real normal data [{title}]", fig_dir / f"{name}_pca.png")
    mp.plot_memorisation(memo["distances"], f"Memorisation check: distance to the closest training record [{title}]", fig_dir / f"{name}_memorisation.png")
    plt.close("all")
    return {
        "validity": gm.validity(synth, real),
        "mean_ks": float(distances["ks_statistic"].mean()), "max_ks": float(distances["ks_statistic"].max()),
        "mean_wasserstein_std_units": float(distances["wasserstein_std_units"].mean()),
        "mean_abs_mean_diff_in_real_std_V": float(moments.drop(index="Amount")["mean_diff_in_real_std"].abs().mean()),
        "mean_std_ratio_V": float(moments.drop(index="Amount")["std_ratio"].mean()),
        "std_ratio_amount": float(moments.loc["Amount", "std_ratio"]),
        "median_central_98_width_ratio": float(quantiles["central_98pct_width_ratio"].median()),
        "median_tail_99_9_width_ratio": float(quantiles["tail_99_9pct_width_ratio"].median()),
        "correlation": {k: v for k, v in corr.items() if not k.endswith("matrix")},
        "pca_explained_variance_ratio": proj["explained_variance_ratio"],
        "c2st": c2st,
        "memorisation": {k: v for k, v in memo.items() if k != "distances"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    data_cfg = load_config(ROOT)
    exp_cfg = load_experiment_config(ROOT, "e1_primary")
    gen_cfg = exp_cfg["generation"]
    hs = exp_cfg["headline_seed"]
    rd = run_dir(ROOT, "e1", hs)
    model = load_trained_model(rd, "cpu")
    tf = FeatureTransformer.load(processed_dir(ROOT, data_cfg) / "transformer_e1.json")

    df = load_raw(ROOT, data_cfg)
    y_all = df[TARGET_COL].to_numpy()
    splits, _ = load_splits(processed_dir(ROOT, data_cfg) / "splits.json")
    parts = derived_partitions(splits, y_all)
    test_normal_idx = splits["test"][y_all[splits["test"]] == 0]   # labels used for partitioning only
    real = df.iloc[test_normal_idx][RAW_INPUT_COLUMNS].reset_index(drop=True)
    real_z = tf.transform(real)
    train_z = tf.transform(df.iloc[parts["e1_train"]])

    n = args.n or gen_cfg["n_samples"] or len(real)
    seed = gen_cfg["seed"] if args.seed is None else args.seed
    metrics_dir, fig_dir = ROOT / "outputs" / "metrics", ROOT / "outputs" / "figures" / "generation"
    syn_dir = ROOT / "data" / "synthetic"
    apply_style()
    base_meta = {"model_run": str(rd.relative_to(ROOT)), "checkpoint_sha256": read_run(rd)["meta"]["checkpoint_sha256"],
                 "sampling_seed": seed, "temperature": 1.0, "transformer": "data/processed/transformer_e1.json"}

    noise_std = fit_decoder_noise_std(model, train_z.astype(np.float32), seed)
    variants = {"decoder_mean": None, "decoder_mean_plus_noise": noise_std}
    summary = {"n_samples": n, "real_reference": f"{len(real):,} normal test transactions (held out)",
               "decoder_noise_std_fitted_on_train": dict(zip(tf.feature_names, noise_std.round(4).tolist())), "variants": {}}
    for name, nstd in variants.items():
        _, synth = generate_synthetic(model, tf, n, seed, temperature=1.0, noise_std=nstd)
        csv = export_synthetic(synth, syn_dir, f"synthetic_e1_{name}_n{n}_seed{seed}",
                               {**base_meta, "sampler": name, "primary": name == "decoder_mean",
                                "noise_std": None if nstd is None else nstd.tolist()})
        summary["variants"][name] = {"csv": str(csv.relative_to(ROOT)),
                                     **evaluate_variant(name, synth, real, real_z, train_z, tf, gen_cfg, seed, metrics_dir, fig_dir)}

    # Reference point: two REAL normal samples (train-normal subsample vs test-normal)
    rng = np.random.default_rng(seed)
    ref_z = train_z[rng.choice(len(train_z), len(real_z), replace=False)]
    ref_d = gm.distance_table(real_z, ref_z, tf.feature_names)
    summary["real_vs_real_reference"] = {"mean_ks": float(ref_d["ks_statistic"].mean()),
                                         "mean_wasserstein_std_units": float(ref_d["wasserstein_std_units"].mean()),
                                         "c2st": gm.classifier_two_sample_test(real_z, ref_z, gen_cfg["c2st_samples"], seed)}

    # Temperature comparison for the primary sampler (summary only)
    temps = []
    for t in gen_cfg["temperatures"]:
        _, s_t = generate_synthetic(model, tf, n, seed, temperature=t)
        d_t = gm.distance_table(real_z, tf.transform(s_t), tf.feature_names)
        temps.append({"temperature": t, "mean_ks": d_t["ks_statistic"].mean(), "mean_wasserstein": d_t["wasserstein_std_units"].mean(),
                      "mean_std_ratio_V": float((s_t.std() / real.std()).drop("Amount").mean())})
    summary["temperatures_decoder_mean"] = temps
    pd.DataFrame(temps).to_csv(metrics_dir / "generation_temperatures.csv", index=False)
    save_json(summary, metrics_dir / "generation_summary.json")

    rows = []
    for name, v in summary["variants"].items():
        rows.append({"sampler": name, "mean_KS": v["mean_ks"], "mean_W1": v["mean_wasserstein_std_units"], "std_ratio_V": v["mean_std_ratio_V"],
                     "tail_width_ratio": v["median_tail_99_9_width_ratio"], "corr_frobenius": v["correlation"]["frobenius_norm_of_difference"],
                     "C2ST_logreg": v["c2st"]["logistic_regression"]["roc_auc_mean"], "C2ST_gboost": v["c2st"]["gradient_boosting"]["roc_auc_mean"],
                     "DCR_median": v["memorisation"]["synthetic_to_train"]["median"]})
    ref = summary["real_vs_real_reference"]
    rows.append({"sampler": "real vs real (reference)", "mean_KS": ref["mean_ks"], "mean_W1": ref["mean_wasserstein_std_units"],
                 "C2ST_logreg": ref["c2st"]["logistic_regression"]["roc_auc_mean"], "C2ST_gboost": ref["c2st"]["gradient_boosting"]["roc_auc_mean"]})
    table = pd.DataFrame(rows).set_index("sampler")
    table.to_csv(metrics_dir / "generation_comparison.csv")
    pd.set_option("display.width", 220)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
