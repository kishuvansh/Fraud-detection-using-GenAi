# VAE Results: Model, Training, Generation and Detection

All numbers come from saved artefacts:
- **Runs:** `outputs/runs/<experiment>/seed_<k>/`
- **Metrics:** `outputs/metrics/`
- **Notebooks:** [03](../notebooks/03_vae_model_and_training.ipynb), [04](../notebooks/04_synthetic_generation.ipynb), [05](../notebooks/05_anomaly_detection_evaluation.ipynb)

Seed 0 was fixed in advance as the headline model. The **5-seed mean ± std** is the representative result.

## 1. Model

- **Network:** a VAE written from scratch in `torch.nn` with 8,813 parameters: encoder 29 → 64 → 32 → (μ, log σ²) in 8 dimensions; decoder 8 → 32 → 64 → 29; LeakyReLU activations; linear output; log σ² clamped to [−10, 10]. The code is in `src/models/vae.py`.
- **Loss:** ½‖x − x̂‖² + β·KL, where the first term is an MSE summed over features (the unit-variance Gaussian negative log-likelihood) and the KL has a closed form. β warms up from 0.1 to 1 over 10 epochs. The code is in `src/training/losses.py`.
- **Configuration:** `configs/e1_primary.yaml` and `configs/e2_unsupervised.yaml`. Hidden sizes, latent size, activation and all training settings are configurable.
- **Checks:** `tests/test_vae.py` (12 tests) covers shapes, rejection of bad input, reparameterization statistics, KL against Monte Carlo (within 1%), gradients reaching every parameter, stability on extreme inputs, overfitting a small batch, and moving between CPU and MPS. `tests/test_training.py` (7 tests) covers bitwise CPU reproducibility, best-epoch selection, threshold calibration and evaluation order.

## 2. Training

| | E1 (normal-only, primary) | E2 (unsupervised, secondary) |
|---|---|---|
| Training rows | 199,021 normal | 199,366 (all, labels ignored) |
| Best epoch (5 seeds) | 168 ± 53 (range 103–246) | 154 ± 41 |
| Runtime per seed (CPU) | 72 s ± 23 s | 66 s ± 17 s |
| Validation reconstruction MSE (z = μ) | 0.408 ± 0.004 | 0.420 ± 0.009 (validation set includes frauds) |
| Active latent dimensions | 8 / 8 in every seed | 8 / 8 |

- **Settings:** Adam (lr 1e-3), batch 512, gradient clipping at norm 5, seeded DataLoaders, deterministic CPU execution.
- **Early stopping:** on the validation negative ELBO, after warm-up, with patience 10. Every run early-stopped before the 300-epoch limit.
- **Epoch limit (deviation D20):** raised from the planned 100 to 300 because seed-0 validation loss was still falling at epoch 100. The decision used only validation loss and was made before any test evaluation.
- **Artefacts:** `checkpoint.pt`, `history.csv`, `train_meta.json` (runtime, versions, device) and `config.yaml` per run. Figures are in `outputs/figures/training/`.

## 3. Threshold calibration (no fraud labels, no test data)

- **Method:** τ is the 99th percentile of reconstruction error on the calibration rows. For E1 these are the 42,647 normal validation transactions; for E2, all 42,721 validation transactions, unlabelled.
- **Values:** E1 seed 0 τ = **1.2292**; across seeds 1.239 ± 0.026. E2: 1.269 ± 0.041.
- **Rationale:** the model only learns normal behaviour, so τ comes from the normal error distribution. The 99th percentile fixes an explicit alert budget of about 1% of normal traffic.
- **Check:** on the unseen test set, the realised false-positive rate is **0.97% ± 0.07%**, so the calibration transfers.

## 4. Synthetic generation (E1 seed 0)

**Sampling and export:**
- **Procedure:** z ~ N(0, I) → decoder → inverse transform → CSV in `data/synthetic/` with 42,647 rows, columns V1–V28 and Amount, and no label.
- **Two samplers, both reported and neither chosen using test data:** the decoder mean (primary) and the decoder mean plus noise fitted on training residuals (pre-listed variant, D21).
- **Real reference:** the 42,647 normal **test** transactions.

| | Decoder mean (primary) | + fitted noise (variant) | Real vs real (noise floor) |
|---|---|---|---|
| Valid rows (finite, Amount ≥ 0, 2 decimals) | 100% | 100% | — |
| Mean per-feature KS statistic | 0.221 | **0.053** | 0.006 |
| Mean Wasserstein-1 (standardised units) | 0.375 | **0.096** | 0.011 |
| Std ratio, synthetic ÷ real (V features) | 0.49 | **0.84** | 1 |
| Tail width ratio (0.1–99.9%) | 0.55 | 0.77 | 1 |
| Correlation-matrix difference (Frobenius) | 3.80 | **1.62** | — |
| Amount–V2 correlation (real −0.52) | **−0.56** | −0.17 | — |
| Classifier two-sample AUC, logistic* | 0.59 (0.58–0.78) | 0.55 | 0.50 |
| Classifier two-sample AUC, gradient boosting | 1.00 | 0.99 | 0.50 |
| Median distance to closest training row (real held-out: 0.89) | 1.89 | 3.10 | — |
| Exact copies of training rows | 0 | 0 | — |

**Assessment:**
- **What works:** the VAE reproduces the centre of the normal distribution (mean error about 0.03 std) and the main linear structure. The decoder mean keeps the Amount–V correlations. All samples are valid and none are memorised copies.
- **Primary sampler is too smooth:** decoder-mean samples have about half the real spread and thin tails, and they sit near an 8-dimensional surface. This is the known "blurry VAE" behaviour.
- **The noise variant fixes the marginals but not the joint structure:** the added noise is independent per feature, so it dilutes real dependencies (Amount–V2 falls to −0.17). Noise in log space also inflates Amount's spread 3.9× after `expm1`.
- **Detectable by a non-linear classifier:** gradient boosting separates both samplers from real data almost perfectly, so the data matches in simple, first-order statistics but not in its full joint structure.
- **\*The linear (logistic) test is unstable** for the decoder-mean sampler. Its AUC is 0.58–0.59 against normal test rows but 0.74–0.78 against normal validation rows (checked over 3 seeds; the real-vs-real control stays at 0.50). A linear classifier on heavy-tailed features is swayed by the rare extreme rows in each reference sample, so the gradient-boosting result (0.99–1.00 on both references) is the reliable conclusion.
- **Lower temperatures shrink spread further:** std ratio 0.32 at 0.8 and 0.15 at 0.5.
- **Interpretability:** samples are vectors in the **anonymised PCA feature space**, not interpretable or verified transaction records.

## 5. Anomaly detection on the unseen test set (42,647 normal, 73 fraud)

**Order of operations:** scores, then the validation threshold, then predictions written to disk with a SHA-256 fingerprint, and only then the test labels. `src/detection/evaluate.py` enforces this, and a test checks it.

| Metric | E1 mean ± std (5 seeds) | E1 seed 0 [95% bootstrap CI] | E2 mean ± std |
|---|---|---|---|
| **Average precision (PR-AUC)** | **0.432 ± 0.047** | 0.492 [0.376, 0.618] | 0.038 ± 0.016 |
| ROC-AUC | 0.947 ± 0.016 | 0.960 [0.926, 0.988] | 0.863 ± 0.016 |
| **Precision** at τ | 0.104 ± 0.013 | 0.118 [0.100, 0.134] | 0.042 ± 0.004 |
| **Recall (TPR)** at τ | 0.652 ± 0.072 | 0.753 [0.644, 0.849] | 0.263 ± 0.025 |
| F1 at τ | 0.179 ± 0.021 | 0.203 [0.174, 0.231] | 0.073 ± 0.006 |
| False-positive rate | 0.0097 ± 0.0007 | 0.0097 [0.009, 0.011] | 0.0102 ± 0.0006 |
| Confusion matrix (TP / FP / FN / TN) | — | 55 / 413 / 18 / 42,234 | — |

A random scorer has an AP of 0.0017. Seed 0 is the best of the five E1 seeds, so the mean is the representative result.

**Alert budgets (E1 seed 0):**
- **Top 100 scores:** 40 frauds (precision 0.40, recall 0.55).
- **Top 500:** 56 frauds (recall 0.77).
- **Top 1,000:** 60 frauds (recall 0.82).

**Threshold sensitivity (post-hoc; thresholds from validation percentiles only; E1 seed 0):**

| Validation percentile | Role | Test precision | Test recall | F1 | FPR | Flagged |
|---|---|---|---|---|---|---|
| 90 | sensitivity | 0.016 | 0.918 | 0.031 | 0.100 | 4,314 |
| 95 | sensitivity | 0.030 | 0.890 | 0.058 | 0.049 | 2,171 |
| 97.5 | sensitivity | 0.055 | 0.822 | 0.103 | 0.024 | 1,094 |
| **99** | **primary** | **0.118** | **0.753** | **0.203** | **0.010** | **468** |
| 99.5 | sensitivity | 0.205 | 0.699 | 0.317 | 0.005 | 249 |
| 99.9 | sensitivity | 0.494 | 0.521 | 0.507 | 0.001 | 77 |

Stricter thresholds give a higher test F1, but adopting one after looking at test results would be test-set tuning. The operating point is an alert-budget decision, so the primary τ stays at p99.

**Secondary scores (E1, average precision):**
- **Negative ELBO:** 0.450 ± 0.040.
- **Importance-weighted −log p(x):** 0.420 ± 0.038.

Both are comparable to reconstruction error, and neither is a fraud probability.

## 6. Interpretation

1. **The one-class VAE detects fraud without fraud labels.** Its AP of 0.43 is about 250× random, and at a 1% alert rate it catches about 65% of frauds. Precision stays low (about 0.10) because fraud is only 0.17% of traffic: false alarms outnumber frauds about 7.5 to 1. That is the base-rate effect, not a calibration failure.
2. **Contamination largely erases detection.** Training on all rows (E2) drops AP from 0.43 to 0.04; E1 is better in every seed (paired bootstrap ΔAP = 0.46, 95% CI 0.35–0.58). The 345 training frauds are 0.17% of rows but carry 4.1% of the squared feature magnitude, so squared-error training learns to reconstruct them. Frauds are also strongly inter-correlated (|r| up to 0.97), so this carries over to unseen frauds: E2's median test-fraud error is 0.86, against 2.08 for E1. Clean normal training data is therefore essential to this method.
3. **Errors.**
   - **Missed frauds** reconstruct well (median error 0.81 against τ = 1.23), although 72% still contain an extreme feature value.
   - **False alarms** are rare but legitimate transactions: higher amounts (median 76.72 against 21.99) and often extreme feature values.
4. **Generation is useful but limited.** The model captures the centre and the main linear structure but not the full spread, tails or non-linear joint structure.

## 7. Limitations

- **Few test frauds:** 73, so confidence intervals are wide and differences between seeds fall within noise.
- **One old dataset:** results come from a single 2-day dataset from 2013 with anonymised features.
- **No supervised comparison:** none was trained, by design; a supervised model would likely be more precise. The VAE's advantage is that it needs no fraud labels to train.
- **Reconstruction error is not a probability.** The likelihood-style scores estimate −log p(x) under the model, not P(fraud | x).
- **Pending work:**
  - architecture sensitivity analyses (latent dimension, β, deterministic-AE baseline, RobustScaler, hour feature, time-ordered split, deduplicated data);
  - the Streamlit app;
  - the consolidated Colab notebook.
