# Generative AI-Based Financial Transaction Modelling Using Variational Autoencoders for Synthetic Data Generation and Anomaly Detection

**Course:** IT429 Generative AI (B.Tech. IT) · **Team:** Yatharth Sachdeva (23/IT/180), Vansh (23/IT/206), Aryan (23/IT/209) · **Guide:** Ms. Deepika

All numbers in this report come from saved artefacts in `outputs/`. They were produced by the repository pipeline: CPU, 5 seeds per experiment, seed 0 fixed in advance as the headline model. The self-contained Colab notebook (`colab/`) re-implements the same pipeline; its verification run is described in `reports/notebook_execution_report.md`.

---

## Abstract

We implement a Variational Autoencoder (VAE) from scratch in PyTorch and apply it to the ULB / Worldline credit-card dataset (284,807 transactions, 0.173% fraud).

- **Generation:** trained only on legitimate transactions, the VAE learns a generative model of normal behaviour and produces synthetic feature vectors by decoding samples from its latent prior.
- **Detection:** the same model flags transactions it reconstructs poorly. On a held-out test set it reaches an average precision (PR-AUC) of **0.432 ± 0.047** (random baseline 0.0017) and a ROC-AUC of **0.947 ± 0.016**, without using any fraud label for training. At a threshold calibrated on normal validation data (about 1% false positives), it catches about 65% of frauds with precision of about 0.10.
- **Contamination:** a fully unsupervised variant trained on all transactions reaches only 0.038 ± 0.016 PR-AUC, showing that even 0.17% fraud contamination largely erases the reconstruction signal.
- **Synthetic data:** samples are valid and not memorised, and they match the centre and main linear structure of normal data. They under-represent its spread, tails and non-linear joint structure.

## 1. Objectives and terminology

1. Build a VAE from scratch: encoder, (μ, log σ²) heads, reparameterization, decoder, and an ELBO objective made of MSE reconstruction plus KL divergence.
2. Generate synthetic transaction feature vectors and evaluate them against real data.
3. Detect anomalous (potentially fraudulent) transactions by reconstruction error, with a threshold calibrated without fraud labels.
4. Compare normal-only training with fully unsupervised training.

**Terminology:** the primary experiment (E1) is **one-class / semi-supervised anomaly detection**. The model never receives labels, but labels are used to select normal training rows. Only the secondary experiment (E2) is **fully unsupervised**.

## 2. Data

**Source:** Kaggle `mlg-ulb/creditcardfraud` (ULB MLG / Worldline), ODbL 1.0 / DbCL 1.0; Dal Pozzolo et al., CIDM 2015.

**Contents:**
- 284,807 transactions over two days in September 2013, of which 492 are fraud.
- Features: `Time`, anonymised PCA components `V1–V28`, `Amount`, and the label `Class`.

**Audit** (`DATASET_AUDIT.md`, notebook 01):
- no missing or infinite values;
- 1,081 exact duplicate rows, and no feature vector carrying both labels;
- `Amount` heavily skewed (skew 16.98, which falls to 0.16 after `log1p`);
- V features centred but not unit-scaled (std 1.96 → 0.33) and heavy-tailed (excess kurtosis up to 933);
- `Time` behaves like a row index with a day/night cycle.

**Preparation** (notebook 02):
- **Split:** stratified group split 70/15/15 (seed 42), with identical feature vectors kept in one split. That gives train 199,366, validation 42,721 and test 42,720 rows; the test set has 73 frauds.
- **Inputs:** `V1–V28` plus `log1p(Amount)`. `Class` and `Time` are never inputs.
- **Scaling:** standardisation fitted only on each experiment's training rows.
- **Checks:** 18 of 18 leakage checks pass.

## 3. Method

**Model:**
- prior $p(z)=\mathcal N(0,I)$;
- encoder $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}\sigma_\phi^2(x))$;
- decoder $p_\theta(x\mid z)=\mathcal N(\hat x_\theta(z),I)$.

**Loss** (negative ELBO):
$$\mathcal L(x)=\tfrac12\lVert x-\hat x\rVert^2+\beta\,\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big),\qquad \mathrm{KL}=-\tfrac12\textstyle\sum_k(1+\log\sigma_k^2-\mu_k^2-\sigma_k^2)$$

**Reparameterization:** $z=\mu+\sigma\odot\varepsilon$, with $\varepsilon\sim\mathcal N(0,I)$.

**Architecture:** 29 → 64 → 32 → 8-dimensional latent (μ and log σ² heads) → 32 → 64 → 29; LeakyReLU activations; linear output; 8,813 parameters.

**Training:**
- Adam (lr 1e-3), batch 512, gradient clipping at norm 5.
- β warms up from 0.1 to 1 over 10 epochs.
- Early stopping on the validation negative ELBO (patience 10, maximum 300 epochs).

**Anomaly score and threshold:**
- **Score:** per-feature reconstruction MSE using z = μ.
- **Threshold:** τ is the 99th percentile of scores on the calibration rows. For E1 these are normal validation rows; for E2, all validation rows, unlabelled.
- **Secondary scores:** the negative ELBO and an importance-weighted estimate of −log p(x) (K = 100).
- **Test protocol:** predictions are written to disk with a fingerprint before any test label is loaded, and the code enforces this order.

**Generation:**
- **Primary sampler:** decoder mean of z ~ N(0, I).
- **Variant:** decoder mean plus Gaussian noise whose per-feature std is estimated from training residuals.

## 4. Results

### 4.1 Anomaly detection (test set: 42,647 normal, 73 fraud)

| Metric | E1 normal-only (primary) | E2 unsupervised | E1 seed 0 [95% bootstrap CI] |
|---|---|---|---|
| **PR-AUC (average precision)** | **0.432 ± 0.047** | 0.038 ± 0.016 | 0.492 [0.376, 0.618] |
| ROC-AUC | 0.947 ± 0.016 | 0.863 ± 0.016 | 0.960 [0.926, 0.988] |
| **Precision** at τ | 0.104 ± 0.013 | 0.042 ± 0.004 | 0.118 [0.100, 0.134] |
| **Recall** at τ | 0.652 ± 0.072 | 0.263 ± 0.025 | 0.753 [0.644, 0.849] |
| F1 at τ | 0.179 ± 0.021 | 0.073 ± 0.006 | 0.203 [0.174, 0.231] |
| False-positive rate | 0.0097 ± 0.0007 | 0.0102 ± 0.0006 | 0.0097 |
| TP / FP / FN / TN (seed 0) | — | 19 / 415 / 54 / 42,232 | 55 / 413 / 18 / 42,234 |

**Reading the table:**
- **Seed 0:** it was fixed in advance and happens to be the best E1 seed, so the mean is the representative result.
- **Alert budgets (E1 seed 0):** the top 100 alerts contain 40 frauds (precision 0.40) and the top 1,000 contain 60 (recall 0.82).
- **Secondary scores (E1):** negative-ELBO PR-AUC 0.450 ± 0.040 and importance-weighted PR-AUC 0.420 ± 0.038, so all three scores rank similarly.

**Threshold sensitivity** (post-hoc; every threshold is a validation percentile):

| Percentile | 90 | 95 | 97.5 | **99 (primary)** | 99.5 | 99.9 |
|---|---|---|---|---|---|---|
| Precision | 0.016 | 0.030 | 0.055 | **0.118** | 0.205 | 0.494 |
| Recall | 0.918 | 0.890 | 0.822 | **0.753** | 0.699 | 0.521 |

Stricter thresholds score better on test F1, but choosing one after seeing test results would be test-set tuning. The operating point should follow an investigation budget.

### 4.2 Secondary experiment and comparison

E1 is better in all 5 seeds on PR-AUC, ROC-AUC, precision, recall and F1. The paired bootstrap gives ΔPR-AUC = 0.46 (95% CI 0.35–0.58) for seed 0.

**Training behaviour:** both models converge and early-stop at similar epochs (168 ± 53 and 154 ± 41). E2 keeps a slightly higher training reconstruction term and a lower KL.

**Score distributions:**
- Normal transactions reconstruct equally well in both models (median error about 0.35).
- The fraud median is **5.7×** the normal median for E1 and only **2.5×** for E2.
- 77% of frauds exceed the normal 99th percentile under E1, against 26% under E2.

**Detection behaviour:** E2 catches 19 frauds and E1 catches 55; 18 of E2's catches are also caught by E1.

**Mechanism:**
- The 345 training frauds are 0.17% of E2's rows but carry about 4.1% of the total squared feature magnitude, so squared-error training learns to reconstruct them.
- Frauds are strongly inter-correlated (|r| up to 0.97), so this transfers to unseen frauds: E2's median test-fraud error is 0.86, against 2.08 for E1.
- **Contamination of the "normal" training distribution therefore directly destroys the reconstruction-error signal.**

### 4.3 Latent space and reconstruction behaviour

**Latent space:**
- E1 maps most test frauds to a compact region away from normal transactions: median latent norm 4.9 against 1.5, and centroid distance 4.7 against a normal spread of about 1.65. It does this without ever having seen a fraud.
- E2 still displaces frauds (centroid distance 3.9). Contamination therefore hurts the *reconstruction* score more than the latent representation, and E2's ELBO-based score (PR-AUC 0.117) beats its reconstruction score.

**Reconstruction behaviour:**
- Fraud reconstruction error is concentrated in a few components: in E1, V27, V14, V17, V12 and V21 supply about 50% of it.
- The E1 decoder pulls extreme fraud values towards the normal manifold, which creates the error.
- Frauds that lie near the normal manifold reconstruct well and are missed.

### 4.4 Synthetic data (E1 seed 0; 42,647 samples per sampler; reference = held-out normal test rows)

| | Decoder mean (primary) | + fitted noise | Real vs real |
|---|---|---|---|
| Valid rows | 100% | 100% | — |
| Mean KS statistic | 0.221 | 0.053 | 0.006 |
| Std ratio (V features) | 0.49 | 0.84 | 1 |
| Correlation difference (Frobenius) | 3.80 | 1.62 | — |
| Amount–V2 correlation (real −0.52) | −0.56 | −0.17 | — |
| Classifier two-sample AUC, logistic* / gradient boosting | 0.59 (0.58–0.78) / 1.00 | 0.55 / 0.99 | 0.50 / 0.50 |
| Median distance to closest training row (real held-out 0.89) | 1.89 | 3.10 | — |

**What the samplers get right:**
- Valid, non-memorised vectors.
- Accurate means (error about 0.03 std).
- The decoder mean keeps the main Amount relationships.

**Where they fall short:**
- **Spread and tails:** decoder-mean samples are too smooth (about half the real spread, thin tails) and lie near an 8-dimensional surface, the familiar "blurry VAE" effect.
- **Noise variant trade-off:** fitted noise repairs the marginals but dilutes dependencies and inflates Amount's spread.
- **Detectability:** both samplers are easy for a non-linear classifier to detect, so the joint distribution is only partly captured.
- **Interpretability:** samples live in anonymised PCA space and carry no labels.

\*The logistic two-sample AUC is unstable for decoder-mean samples: 0.58–0.59 against normal test rows, 0.74–0.78 against normal validation rows (3 seeds; real vs real stays at 0.50). Heavy-tailed outliers sway the linear classifier, so the stable gradient-boosting result carries the conclusion. The Colab notebook uses the validation reference, so it reports the higher value.

## 5. Discussion

1. **The generative approach works without fraud labels.** A VAE trained only on legitimate behaviour ranks fraud about 250× better than chance, and its latent space separates fraud it has never seen.
2. **Class imbalance dominates precision.** At a 0.97% false-positive rate and 75% recall (seed 0), precision is only 0.12, about 7.5 false alarms per fraud caught. The same detector would reach about 0.44 precision at a 1% fraud rate and 0.90 at 10%. PR-AUC is the right primary metric; ROC-AUC looks optimistic.
3. **Clean normal data is essential.** The E1/E2 gap is the most important practical finding: reconstruction detectors are fragile to contamination, and labels used only to *select* clean training data make a large difference.
4. **Reconstruction error is a useful but imperfect anomaly signal.** It measures distance from learned normal behaviour, not the probability of fraud, and it misses frauds that look normal.
5. **Generation quality is moderate.** The model learns first-order structure well and higher-order structure poorly, which is typical for a small Gaussian-decoder VAE on heavy-tailed data.

## 6. Limitations

- **Old, short data:** the dataset covers **September 2013 and only two days**. Results **cannot automatically be generalised to contemporary banking systems**.
- **Anonymised features:** `V1–V28` are **anonymised PCA components**, so features cannot be interpreted, and the providers' PCA was fitted on all rows.
- **Synthetic samples:** they exist in the **anonymised feature space**. They are not interpretable or verified transactions, and they carry no labels.
- **Missed frauds:** reconstruction error **does not guarantee that every fraudulent transaction will be detected**; frauds resembling normal behaviour are missed.
- **Protocol:** the primary normal-only procedure is a **one-class / semi-supervised** anomaly-detection set-up, not strictly fully unsupervised learning.
- **Small test set:** only 73 test frauds, so confidence intervals are wide and differences between seeds are within noise.
- **Fixed hyperparameters:** they were set in advance. Sensitivity to latent size, β, scaling, a deterministic-AE baseline, time-ordered splits and deduplication has not been run yet.
- **Generator limits:** a unit-variance Gaussian decoder and an 8-dimensional latent space limit sample sharpness and joint fidelity.
- **No supervised baseline:** none was trained, by design. A supervised classifier would likely be more precise, but it needs fraud labels.

## 7. Future work

- **Better scores:** latent- and KL-aware scores (ELBO, latent density), which held up better under contamination.
- **Contamination-robust training:** robust losses, or iterative trimming of high-error rows.
- **Richer decoders and priors:** learned per-feature variance, heavy-tailed likelihoods, flow priors, β-VAE / VQ-VAE; conditional VAEs for controlled generation.
- **Planned sensitivity analyses,** plus drift studies with time-ordered splits.
- **Newer data:** validation on recent, interpretable and longer-horizon transaction datasets.

## 8. Reproducibility and deliverables

**Reproducibility:**
- **Fixed seeds:** split 42; models 0–4; validation noise 1234; scores 2024; bootstrap 7; generation 0.
- **Determinism:** bitwise-deterministic CPU training; dataset SHA-256 checked on load; 26 automated tests.
- **Deviations from the plan:** D20 (maximum epochs 100 → 300, decided from validation loss only) and D21 (generation noise variant reported alongside the primary sampler).

**Deliverables:**

| Deliverable | Location |
|---|---|
| Self-contained Colab notebook | `colab/VAE_Financial_Transaction_Modeling_Colab.ipynb` |
| Phase notebooks | `notebooks/01`–`06` |
| Trained checkpoints (primary E1, secondary E2) | `outputs/runs/{e1,e2}/seed_*/checkpoint.pt` |
| Synthetic CSVs | `data/synthetic/` |
| Metrics | `outputs/metrics/` (e.g. `detection_per_seed.csv`, `comparison_table.csv`, `generation_comparison.csv`) |
| Figures | `outputs/figures/{eda,training,generation,detection,analysis}/` |
