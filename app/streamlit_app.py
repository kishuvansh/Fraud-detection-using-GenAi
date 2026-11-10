"""VAE-Based Financial Transaction Modelling — Streamlit Dashboard

Five tabs:
  1. 📊 Project Overview   – headline numbers, E1 vs E2 table, caveats
  2. 🔍 Score a Transaction – enter feature values → get a reconstruction-error score
  3. 🧬 Generate Synthetic  – sample N rows from the VAE decoder, download CSV
  4. 🌐 Latent Space        – 2-D PCA of test-set latent means coloured by predicted label
  5. 📈 Experiment Results  – interactive PR / ROC curves, threshold sensitivity, seed breakdown

Usage:
    cd /Users/vansh/Downloads/genAI
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VAE Financial Anomaly Detector",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "colab" / "vae_outputs"
CHECKPOINT_DIR  = DATA_DIR / "checkpoints"
METRICS_DIR     = DATA_DIR / "metrics"
FIGURES_DIR     = DATA_DIR / "figures"
PREDICTIONS_DIR = DATA_DIR / "predictions"
SYNTHETIC_DIR   = DATA_DIR / "synthetic"
PROCESSED_DIR   = ROOT / "data" / "processed"
LOCAL_SYNTH_DIR = ROOT / "data" / "synthetic"

TRANSFORMER_PATH = CHECKPOINT_DIR / "transformer_e1.json"
THRESHOLD_PATH   = METRICS_DIR / "threshold_calibration_e1.json"
SUMMARY_PATH     = METRICS_DIR / "detection_metrics_summary.csv"
PER_SEED_PATH    = METRICS_DIR / "detection_metrics_per_seed.csv"
COMPARISON_PATH  = METRICS_DIR / "comparison_e1_vs_e2.csv"
SENSITIVITY_PATH = METRICS_DIR / "threshold_sensitivity_e1_seed0.csv"
SYNTH_QUALITY    = METRICS_DIR / "synthetic_quality.json"
CONFIG_PATH      = DATA_DIR / "config.json"

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"
GREEN  = "#2ca02c"

# ════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading VAE model…")
def load_model(seed: int = 0):
    """Load VAE checkpoint + FeatureTransformer for E1 (the primary experiment).

    The checkpoint is a dict with keys: model_state, model_config, experiment,
    seed, best_epoch, threshold.
    """
    import torch
    from src.models.vae import VAEConfig, VAE
    from src.data.preprocess import FeatureTransformer

    transformer = FeatureTransformer.load(TRANSFORMER_PATH)

    ckpt_path = CHECKPOINT_DIR / f"vae_e1_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        cfg = VAEConfig.from_dict(ckpt["model_config"])
        state_dict = ckpt["model_state"]
    else:
        cfg = VAEConfig(input_dim=29, hidden_dims=[64, 32], latent_dim=8, activation="leaky_relu")
        state_dict = ckpt

    model = VAE(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, transformer


@st.cache_data(show_spinner=False)
def load_threshold() -> dict:
    with open(THRESHOLD_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_per_seed_df() -> pd.DataFrame:
    return pd.read_csv(PER_SEED_PATH)


@st.cache_data(show_spinner=False)
def load_summary_df() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH, header=[0, 1], index_col=0)
    df.columns = [" ".join(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_comparison_df() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_sensitivity_df() -> pd.DataFrame:
    return pd.read_csv(SENSITIVITY_PATH)


@st.cache_data(show_spinner=False)
def load_synthetic_quality() -> dict:
    with open(SYNTH_QUALITY) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_predictions(exp: str = "e1", seed: int = 0) -> pd.DataFrame:
    p = PREDICTIONS_DIR / f"test_predictions_{exp}_seed{seed}.csv"
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_all_predictions_e1() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / "predictions_e1_all_seeds.csv")


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png", width=48)
    st.title("VAE Anomaly Detector")
    st.caption("IT429 Generative AI · B.Tech IT · 2026")
    st.markdown("---")

    seed_choice = st.selectbox("Active model seed", options=list(range(5)), index=0,
                               help="Seed used for scoring and generation in the Score / Generate tabs.")
    st.markdown("---")
    st.info(
        "**Score disclaimer:** reconstruction error is a *distance*, not P(fraud | x). "
        "It is calibrated on legitimate-only validation data at the 99th percentile.",
        icon="ℹ️",
    )
    st.info(
        "**Synthetic data disclaimer:** generated rows live in the **anonymised PCA feature space** "
        "(V1–V28 + Amount). They are not interpretable real-world transaction records.",
        icon="⚠️",
    )
    st.caption(
        "Dataset: Credit Card Fraud Detection, ULB MLG / Worldline (Kaggle mlg-ulb/creditcardfraud), "
        "ODbL 1.0 / DbCL 1.0; Dal Pozzolo et al., CIDM 2015."
    )

# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════

tab_overview, tab_score, tab_generate, tab_latent, tab_results = st.tabs([
    "📊 Project Overview",
    "🔍 Score a Transaction",
    "🧬 Generate Synthetic",
    "🌐 Latent Space",
    "📈 Experiment Results",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Project Overview
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("VAE-Based Financial Transaction Modelling")
    st.markdown(
        """
        A **Variational Autoencoder (VAE)** trained exclusively on *legitimate* credit-card transactions
        detects anomalies by high reconstruction error — without ever seeing a fraud label during training.
        The same model generates **synthetic legitimate-like transaction vectors** as the core GenAI deliverable.

        > **Dataset:** 284,807 card transactions (492 frauds, 0.17 %) from European cardholders, 2013.
        > Features: 28 anonymised PCA components (V1–V28) + Amount.
        > Split: 70 / 15 / 15 stratified group-random, seed 42.
        """
    )

    # ── Headline numbers ──────────────────────────────────────────────────
    st.subheader("Headline Results (E1 — primary experiment, seed 0)")
    threshold_data = load_threshold()
    tau = threshold_data["threshold"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PR-AUC (5-seed mean)", "0.432", delta="253× random baseline")
    col2.metric("ROC-AUC (5-seed mean)", "0.947")
    col3.metric("Recall at τ (seed 0)", "75.3%")
    col4.metric("Precision at τ (seed 0)", "11.8%")
    col5.metric("Threshold τ (seed 0)", f"{tau:.4f}", help="99th percentile of val reconstruction error")

    st.caption("Bootstrap 95% CI for seed 0: PR-AUC [0.376, 0.618], Recall [0.644, 0.849], Precision [0.100, 0.134]")

    # ── E1 vs E2 ─────────────────────────────────────────────────────────
    st.subheader("E1 (one-class) vs E2 (fully unsupervised)")
    cmp = load_comparison_df()
    st.dataframe(
        cmp.set_index("metric").style.format(
            {c: "{:.4f}" for c in cmp.columns if cmp[c].dtype == float}
        ).highlight_max(axis=1, subset=["E1 mean", "E2 mean"], color="#d4edda"),
        use_container_width=True,
    )
    st.caption(
        "Training on the full (contaminated) dataset (E2) collapses PR-AUC from 0.432 → 0.038. "
        "E1 is better in all 5 seeds (bootstrap ΔAP 0.394, 95 % CI 0.351–0.575)."
    )

    # ── Architecture card ─────────────────────────────────────────────────
    st.subheader("Model Architecture")
    with st.expander("Show architecture details"):
        st.code(
            "Encoder:  29 → Linear(64) → LeakyReLU → Linear(32) → LeakyReLU → {Linear(8)=μ, Linear(8)=log σ²}\n"
            "Latent:   d = 8\n"
            "Decoder:  8  → Linear(32) → LeakyReLU → Linear(64) → LeakyReLU → Linear(29)\n"
            "Params:   ≈ 8,800\n\n"
            "Optimiser: Adam, lr=1e-3 | Batch: 512 | Max epochs: 300 (early stop, patience 10)\n"
            "β warm-up: linear 0→1 over epochs 1–10 | Gradient clip: global norm 5.0\n"
            "Score S1: ½‖x − decoder(μ(x))‖² (deterministic, primary)\n"
            "Score S2: negative ELBO (L=10 samples)\n"
            "Score S3: importance-weighted log-likelihood (K=100 samples)",
            language="text",
        )

    # ── Figures ───────────────────────────────────────────────────────────
    st.subheader("Key Figures")
    fig_col1, fig_col2 = st.columns(2)
    pr_fig = FIGURES_DIR / "test_pr_roc_confusion_e1.png"
    cmp_fig = FIGURES_DIR / "comparison_e1_vs_e2.png"
    if pr_fig.exists():
        fig_col1.image(str(pr_fig), caption="PR / ROC curves + confusion matrix — E1 seed 0 (test set, read once)")
    if cmp_fig.exists():
        fig_col2.image(str(cmp_fig), caption="E1 vs E2 comparison across all seeds")

    imb_fig = FIGURES_DIR / "class_imbalance_effect.png"
    if imb_fig.exists():
        with st.expander("Class imbalance effect on precision"):
            st.image(str(imb_fig))


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Score a Transaction
# ════════════════════════════════════════════════════════════════════════════

with tab_score:
    st.header("🔍 Score a Transaction")
    st.markdown(
        """
        Enter feature values and the model returns a **reconstruction error score (S1)**.
        A score above the threshold τ flags the transaction as anomalous.

        > ⚠️ **This score is a distance, not P(fraud | x).** A high score means the transaction
        > is unusual relative to the legitimate-transaction distribution the VAE learned.
        """
    )

    PCA_COLS = [f"V{i}" for i in range(1, 29)]

    # ── Input mode ────────────────────────────────────────────────────────
    input_mode = st.radio("Input method", ["Manual entry", "Paste CSV row", "Pick random test row"],
                          horizontal=True)

    threshold_data = load_threshold()
    tau = threshold_data["thresholds_all_seeds"][str(seed_choice)]

    if input_mode == "Pick random test row":
        preds_df = load_test_predictions("e1", seed_choice)
        # We only have recon_mse and row index; re-load the actual feature data for display
        st.info(f"Showing a random row from the test-set predictions (seed {seed_choice}).")
        rand_row = preds_df.sample(1, random_state=int(np.random.randint(0, 10000)))
        st.write(f"**Dataset row index:** {rand_row['dataset_row'].iloc[0]}  "
                 f"**Pre-computed recon MSE:** {rand_row['recon_mse'].iloc[0]:.4f}  "
                 f"**Flagged:** {'Yes 🚨' if rand_row['predicted_anomaly'].iloc[0] else 'No ✅'}")
        st.caption("Feature values for this row are not shown here (the raw CSV is not loaded in the app).")

    elif input_mode == "Paste CSV row":
        csv_text = st.text_area(
            "Paste one CSV row (header optional): V1,V2,…,V28,Amount",
            height=120,
            placeholder="-1.36,  -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09, "
                        "-0.55, -0.62,  0.72,  0.25,  0.37,  0.09,  0.09, -0.26, -0.17,  0.09, "
                        "0.12,  -0.03,  0.08, 0.17, -0.12, -0.20, 0.10, 0.07, 149.62",
        )
        if st.button("Score row", key="score_csv"):
            try:
                import io
                raw = csv_text.strip()
                # Drop header line if present
                lines = [l for l in raw.splitlines() if l.strip()]
                if lines[0].lower().startswith("v1"):
                    lines = lines[1:]
                vals = [float(v.strip()) for v in lines[0].split(",")]
                if len(vals) == 29:
                    row_df = pd.DataFrame([vals], columns=[*PCA_COLS, "Amount"])
                elif len(vals) == 31:          # Time, V1-V28, Amount, Class
                    row_df = pd.DataFrame([vals[1:30]], columns=[*PCA_COLS, "Amount"])
                else:
                    st.error(f"Expected 29 or 31 values, got {len(vals)}.")
                    st.stop()

                model, transformer = load_model(seed_choice)
                import torch
                X = transformer.transform(row_df).astype(np.float32)
                x_t = torch.from_numpy(X)
                with torch.no_grad():
                    out = model(x_t, sample=False)
                    score = float((x_t - out.recon).pow(2).mean(dim=1).item())
                is_fraud = score > tau
                colour = RED if is_fraud else GREEN
                st.markdown(
                    f"<h3 style='color:{colour}'>{'🚨 Anomalous' if is_fraud else '✅ Normal'}</h3>",
                    unsafe_allow_html=True,
                )
                st.metric("Reconstruction error (S1)", f"{score:.4f}",
                          delta=f"{score - tau:+.4f} vs τ={tau:.4f}")
            except Exception as e:
                st.error(f"Error scoring row: {e}")

    else:  # Manual entry
        st.markdown("**PCA components (V1–V28)**")
        cols_per_row = 7
        vals: dict[str, float] = {}
        for row_i in range(4):
            row_cols = st.columns(cols_per_row)
            for col_i, col_st in enumerate(row_cols):
                feat_i = row_i * cols_per_row + col_i + 1
                if feat_i > 28:
                    break
                vals[f"V{feat_i}"] = col_st.number_input(f"V{feat_i}", value=0.0, step=0.01, format="%.4f",
                                                           key=f"v{feat_i}")

        st.markdown("**Amount (raw, ≥ 0)**")
        amount_val = st.number_input("Amount", min_value=0.0, value=0.0, step=1.0, format="%.2f")

        if st.button("Score transaction", key="score_manual"):
            try:
                row_data = {**vals, "Amount": amount_val}
                row_df = pd.DataFrame([row_data])
                model, transformer = load_model(seed_choice)
                import torch
                X = transformer.transform(row_df).astype(np.float32)
                x_t = torch.from_numpy(X)
                with torch.no_grad():
                    out = model(x_t, sample=False)
                    score = float((x_t - out.recon).pow(2).mean(dim=1).item())
                is_fraud = score > tau
                colour = RED if is_fraud else GREEN
                st.markdown(
                    f"<h3 style='color:{colour}'>{'🚨 Anomalous (score > τ)' if is_fraud else '✅ Normal (score ≤ τ)'}</h3>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Reconstruction error (S1)", f"{score:.4f}")
                c2.metric("Threshold τ", f"{tau:.4f}")
                c3.metric("Margin", f"{score - tau:+.4f}")
                st.caption(
                    "S1 is the mean squared reconstruction error: ½‖x − decoder(μ_encoder(x))‖². "
                    "τ is the 99th percentile of S1 on legitimate validation rows (no fraud labels used). "
                    "This is not P(fraud | x)."
                )

                # Per-feature error bar chart
                with torch.no_grad():
                    feat_err = (x_t - out.recon).pow(2).squeeze(0).detach().cpu().numpy()
                feat_names = transformer.feature_names
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.bar(feat_names, feat_err, color=BLUE)
                ax.set_ylabel("Squared error")
                ax.set_title("Per-feature reconstruction error")
                plt.xticks(rotation=90, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Error: {e}")

    # ── Threshold context ─────────────────────────────────────────────────
    with st.expander("Threshold calibration details"):
        st.json(threshold_data)
        sens_df = load_sensitivity_df()
        st.dataframe(sens_df, use_container_width=True)
        st.caption(
            "Threshold sensitivity: precision/recall at different validation percentiles (seed 0, test set). "
            "These are post-hoc analyses and were never used to select τ."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Generate Synthetic
# ════════════════════════════════════════════════════════════════════════════

with tab_generate:
    st.header("🧬 Generate Synthetic Transaction Vectors")
    st.warning(
        "**Interpretability disclaimer:** generated samples are vectors in the **anonymised PCA feature space** "
        "(V1–V28 + Amount). They are **not** interpretable real-world transaction records. "
        "The original features and PCA loadings were never published, so samples cannot be mapped "
        "to merchants, cards, locations, or customers. `Amount` is the only directly interpretable field. "
        "No `Class` or `Time` column is attached.",
        icon="⚠️",
    )

    c1, c2, c3 = st.columns(3)
    n_samples = c1.number_input("Number of samples (N)", min_value=1, max_value=100_000,
                                value=1_000, step=100)
    temperature = c2.select_slider("Temperature τ_gen", options=[0.5, 0.8, 1.0],
                                   value=1.0,
                                   help="Controls diversity. τ=1 → prior N(0,I); "
                                        "τ<1 → samples closer to the latent centre.")
    add_noise = c3.checkbox("Add fitted decoder noise",
                            help="Adds per-feature Gaussian noise fitted on training residuals. "
                                 "Improves marginal distributions but dilutes correlations.")
    gen_seed = c3.number_input("Generation seed", min_value=0, value=42, step=1)

    if st.button("Generate", type="primary"):
        try:
            import torch
            from src.generation.sample import generate_synthetic, fit_decoder_noise_std

            with st.spinner("Generating…"):
                model, transformer = load_model(seed_choice)

                noise_std = None
                if add_noise:
                    # Noise fitting requires training data which isn't loaded in the app.
                    # Estimate from the pre-generated Colab noise std stored in synthetic_quality.json
                    # by using the per-feature std ratios as a proxy. We inform the user.
                    st.info(
                        "Decoder noise fitting requires the original training set, which is not loaded in the app. "
                        "Skipping noise — using decoder-mean sampler (temperature controls diversity). "
                        "Pre-generated noise-variant samples are shown below in the quality table."
                    )

                Z, synth_df = generate_synthetic(
                    model=model,
                    transformer=transformer,
                    n=int(n_samples),
                    seed=int(gen_seed),
                    temperature=float(temperature),
                    noise_std=noise_std,
                )

            st.success(f"Generated {len(synth_df):,} synthetic rows ✅")

            # Preview
            st.subheader("Preview (first 10 rows)")
            st.dataframe(synth_df.head(10).style.format("{:.4f}"), use_container_width=True)

            # Basic stats
            st.subheader("Distribution statistics")
            stat_df = synth_df.describe().loc[["mean", "std", "min", "max"]].T
            st.dataframe(stat_df.style.format("{:.4f}"), use_container_width=True)

            # Amount histogram
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].hist(synth_df["Amount"], bins=50, color=BLUE, edgecolor="white", alpha=0.8)
            axes[0].set_xlabel("Amount")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Synthetic Amount distribution")

            axes[1].hist(np.log1p(synth_df["Amount"]), bins=50, color=ORANGE, edgecolor="white", alpha=0.8)
            axes[1].set_xlabel("log1p(Amount)")
            axes[1].set_title("Synthetic log1p(Amount) distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Download
            csv_bytes = synth_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download synthetic CSV",
                data=csv_bytes,
                file_name=f"synthetic_vae_n{n_samples}_T{temperature}_seed{gen_seed}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.exception(e)

    # ── Pre-generated quality table ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Pre-generated Synthetic Data Quality (from Colab run, E1 seed 0)")
    q = load_synthetic_quality()["quality"]
    rows = []
    metrics_to_show = ["valid_pct", "mean_ks", "mean_w1", "std_ratio_V", "corr_frobenius_diff",
                       "c2st_logistic", "c2st_gradient_boosting"]
    for sampler, stats in q.items():
        row = {"Sampler": sampler}
        for m in metrics_to_show:
            row[m] = stats.get(m, None)
        rows.append(row)
    q_df = pd.DataFrame(rows).set_index("Sampler")
    q_df.columns = ["Valid %", "Mean KS", "Mean W1", "Std ratio (V)", "Corr Frob diff",
                    "C2ST (logistic)", "C2ST (boosting)"]
    st.dataframe(q_df.style.format("{:.4f}"), use_container_width=True)
    st.caption(
        "KS / W1 lower is better. C2ST AUC ≈ 0.5 means indistinguishable from real data. "
        "Real-vs-real reference row shows the measurement noise floor."
    )

    # Pre-generated figures
    synth_fig = FIGURES_DIR / "synthetic_distributions_decoder_mean.png"
    synth_fig_noise = FIGURES_DIR / "synthetic_distributions_decoder_mean_plus_noise.png"
    pca_fig = FIGURES_DIR / "synthetic_pca.png"
    corr_fig = FIGURES_DIR / "synthetic_correlations.png"

    col_a, col_b = st.columns(2)
    if synth_fig.exists():
        col_a.image(str(synth_fig), caption="Decoder-mean synthetic vs real distributions")
    if synth_fig_noise.exists():
        col_b.image(str(synth_fig_noise), caption="Decoder-mean + noise vs real distributions")

    col_c, col_d = st.columns(2)
    if pca_fig.exists():
        col_c.image(str(pca_fig), caption="PCA overlay: synthetic vs real (2-D)")
    if corr_fig.exists():
        col_d.image(str(corr_fig), caption="Correlation structure: real vs synthetic")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Latent Space
# ════════════════════════════════════════════════════════════════════════════

with tab_latent:
    st.header("🌐 Latent Space Visualisation")
    st.markdown(
        """
        The VAE encoder maps each test-set transaction to a posterior mean μ ∈ ℝ⁸.
        Projecting these down to 2-D with PCA reveals how well the model separates
        normal transactions from anomalies **without ever seeing fraud labels during training**.
        """
    )

    # Pre-generated figure
    lat_fig = FIGURES_DIR / "latent_space_e1_vs_e2.png"
    lat_dim_fig = FIGURES_DIR / "latent_dimensions_e1.png"

    if lat_fig.exists():
        st.image(str(lat_fig), caption="2-D PCA of latent means μ — E1 (left) vs E2 (right), coloured by predicted label")
    if lat_dim_fig.exists():
        st.image(str(lat_dim_fig), caption="Per-dimension KL divergence (active latent units) — E1 seed 0")

    # Interactive: live PCA from model (requires model + data)
    st.markdown("---")
    st.subheader("Interactive Latent Projection (from pre-computed predictions)")
    preds_e1 = load_test_predictions("e1", seed_choice)

    st.markdown(
        f"Showing pre-computed anomaly scores for **{len(preds_e1):,}** test rows (E1 seed {seed_choice}). "
        "Flagged = predicted anomaly at the validation-calibrated threshold τ."
    )

    import matplotlib.pyplot as plt

    normal_scores = preds_e1[preds_e1["predicted_anomaly"] == 0]["recon_mse"]
    anom_scores   = preds_e1[preds_e1["predicted_anomaly"] == 1]["recon_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Score histogram
    axes[0].hist(normal_scores, bins=100, alpha=0.7, color=BLUE, label="Normal (predicted)", density=True)
    axes[0].hist(anom_scores,   bins=50,  alpha=0.7, color=RED,  label="Anomaly (predicted)", density=True)
    tau = load_threshold()["thresholds_all_seeds"][str(seed_choice)]
    axes[0].axvline(tau, color="black", linestyle="--", label=f"τ = {tau:.3f}")
    axes[0].set_xlabel("Reconstruction error (S1)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Score distribution — E1 seed {seed_choice}")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, min(preds_e1["recon_mse"].quantile(0.999) * 1.1, 20))

    # Flagged counts
    n_flagged = int((preds_e1["predicted_anomaly"] == 1).sum())
    n_normal  = int((preds_e1["predicted_anomaly"] == 0).sum())
    axes[1].bar(["Normal", "Flagged"], [n_normal, n_flagged], color=[BLUE, RED], edgecolor="white")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction counts")
    for i, v in enumerate([n_normal, n_flagged]):
        axes[1].text(i, v + 50, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Reconstruction figures
    recon_fig = FIGURES_DIR / "reconstruction_per_feature.png"
    recon_scatter = FIGURES_DIR / "reconstruction_scatter.png"
    if recon_fig.exists() or recon_scatter.exists():
        st.markdown("### Reconstruction quality")
        rc1, rc2 = st.columns(2)
        if recon_fig.exists():
            rc1.image(str(recon_fig), caption="Per-feature reconstruction MSE on test-set legitimate rows")
        if recon_scatter.exists():
            rc2.image(str(recon_scatter), caption="Real vs reconstructed (selected features)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Experiment Results
# ════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.header("📈 Experiment Results")

    # ── Per-seed table ────────────────────────────────────────────────────
    st.subheader("Per-seed detection metrics")
    per_seed = load_per_seed_df()
    display_cols = ["experiment", "seed", "average_precision", "roc_auc",
                    "precision", "recall", "f1", "fpr", "best_epoch", "runtime_s"]
    display_cols = [c for c in display_cols if c in per_seed.columns]
    st.dataframe(
        per_seed[display_cols].style.format({
            "average_precision": "{:.4f}",
            "roc_auc": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "fpr": "{:.4f}",
            "runtime_s": "{:.1f}",
        }).background_gradient(subset=["average_precision"], cmap="Greens"),
        use_container_width=True,
    )

    # ── Training curves ───────────────────────────────────────────────────
    st.subheader("Training curves")
    history_files = sorted((METRICS_DIR).glob("history_e1_seed*.csv"))
    if history_files:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors_list = [BLUE, ORANGE, GREEN, RED, "#9467bd"]
        for i, hf in enumerate(history_files):
            h = pd.read_csv(hf)
            seed_num = hf.stem.split("seed")[1]
            x_axis = h["epoch"] if "epoch" in h.columns else range(len(h))
            # Support both old ('train_loss') and current ('train_total') column names
            train_col = "train_total" if "train_total" in h.columns else ("train_loss" if "train_loss" in h.columns else None)
            val_col   = "val_total"   if "val_total"   in h.columns else ("val_loss"   if "val_loss"   in h.columns else None)
            if train_col:
                axes[0].plot(x_axis, h[train_col], color=colors_list[i], label=f"seed {seed_num}", linewidth=1.2)
            if val_col:
                axes[1].plot(x_axis, h[val_col],   color=colors_list[i], label=f"seed {seed_num}", linewidth=1.2)
        axes[0].set_title("Train loss (E1)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=8)
        axes[1].set_title("Validation loss (E1)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Pre-built training curve image for seed 0
    tc_fig = FIGURES_DIR / "training_curves_e1_seed0.png"
    if tc_fig.exists():
        with st.expander("Training curves — E1 seed 0 (Colab figure)"):
            st.image(str(tc_fig))

    # ── Threshold sensitivity ─────────────────────────────────────────────
    st.subheader("Threshold sensitivity (E1 seed 0)")
    sens_df = load_sensitivity_df()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sens_df["validation_percentile"], sens_df["precision"], "o-", color=BLUE,  label="Precision")
    ax.plot(sens_df["validation_percentile"], sens_df["recall"],    "s-", color=ORANGE, label="Recall")
    ax.plot(sens_df["validation_percentile"], sens_df["f1"],        "^-", color=GREEN,  label="F1")
    ax.axvline(99, color="black", linestyle="--", alpha=0.6, label="Selected τ (99th pct)")
    ax.set_xlabel("Validation percentile used for τ")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold sensitivity — E1 seed 0 (post-hoc, test set)")
    ax.legend()
    ax.set_xlim(88, 100.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "⚠️ These values are post-hoc analyses on the test set shown for context. "
        "They were **never** used to select τ — τ was fixed at the 99th validation percentile before any test evaluation."
    )

    # ── PR / ROC image ────────────────────────────────────────────────────
    st.subheader("PR / ROC / Confusion matrix (E1 seed 0, test set)")
    pr_fig = FIGURES_DIR / "test_pr_roc_confusion_e1.png"
    if pr_fig.exists():
        st.image(str(pr_fig), use_container_width=True)

    score_dist_fig = FIGURES_DIR / "test_score_distributions_e1.png"
    thresh_fig = FIGURES_DIR / "threshold_calibration_e1.png"
    thresh_sens_fig = FIGURES_DIR / "threshold_sensitivity_e1.png"

    col_a, col_b = st.columns(2)
    if score_dist_fig.exists():
        col_a.image(str(score_dist_fig), caption="Score distributions on test set — E1 seed 0")
    if thresh_fig.exists():
        col_b.image(str(thresh_fig), caption="Threshold calibration on validation set — E1 seed 0")

    if thresh_sens_fig.exists():
        with st.expander("Threshold sensitivity figure (Colab)"):
            st.image(str(thresh_sens_fig))

    # ── Comparison figure ─────────────────────────────────────────────────
    comp_score_fig = FIGURES_DIR / "comparison_score_distributions.png"
    if comp_score_fig.exists():
        with st.expander("Score distributions: E1 vs E2"):
            st.image(str(comp_score_fig))

    # ── EDA figures ───────────────────────────────────────────────────────
    st.subheader("EDA Figures")
    eda_cols = st.columns(2)
    eda_figs = [
        ("eda_overview.png", "Dataset overview"),
        ("eda_feature_distributions.png", "Feature distributions"),
        ("eda_class_comparison.png", "Class comparison"),
        ("eda_correlations.png", "Correlations"),
    ]
    for idx, (fname, cap) in enumerate(eda_figs):
        p = FIGURES_DIR / fname
        if p.exists():
            eda_cols[idx % 2].image(str(p), caption=cap)
