# Project Plan — VAE-Based Financial Transaction Modelling

**Working title:** Generative AI-Based Financial Transaction Modeling Using Variational Autoencoders for Synthetic Data Generation and Unsupervised Anomaly Detection (title kept; terminology clarified in the report, see §13 A2)
**Course:** IT429 Generative AI (B.Tech. IT) · **Team:** Yatharth Sachdeva (23/IT/180), Vansh (23/IT/206), Aryan (23/IT/209) · **Guide:** Ms. Deepika
**Version:** 2.0 (finalised plan), 2026-11-06
**Status:** Planning complete and all owner decisions resolved (§13). No code written and no models trained yet.
**Companion documents:** [DATASET_AUDIT.md](DATASET_AUDIT.md) (all dataset statistics come from it) · [DATASET_COMPARISON.md](DATASET_COMPARISON.md) (why the Mendeley alternative was rejected)

---

## 0. Project folder (as of this plan)

| File | Notes |
|---|---|
| `creditcard.csv` | **Primary dataset. Immutable and read-only.** SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| `Project Proposal_ Variational Autoencoder (VAE) for Unsupervised Financial Anomaly Detection.pdf` | Original proposal (2 pages) |
| `PROJECT_PLAN.md`, `DATASET_AUDIT.md`, `DATASET_COMPARISON.md` | Planning documents |

No notebooks, scripts, or experimental results exist yet.

**Local environment (checked, nothing installed):** Python 3.13.5, PyTorch 2.10.0 (MPS available), pandas 2.3.1, NumPy 2.3.2, scikit-learn 1.7.1, SciPy 1.16.1, Matplotlib 3.10.5, seaborn 0.13.2, Streamlit 1.58.0, PyYAML, tqdm. `ipykernel` is missing (see §13 A4). Hardware: Mac M4, 16 GB RAM, about 47 GB free disk.

---

## 1. Dataset selection (formal)

**Selected dataset:** the ULB / Worldline *Credit Card Fraud Detection* dataset, `creditcard.csv`.

| Item | Value |
|---|---|
| Official source | Kaggle `mlg-ulb/creditcardfraud`, v3 (2018-03-23), owner: Machine Learning Group, ULB |
| Content | 284,807 card transactions from European cardholders over 2 days in September 2013; 492 frauds (0.1727%) |
| Columns | `Time`, `V1`–`V28` (PCA components), `Amount`, `Class` |
| Local file identity | Byte size matches Kaggle exactly (150,828,752 bytes); SHA-256 recorded above |
| Licence | Kaggle lists **ODbL 1.0 (database) / DbCL 1.0 (contents)**. Open ambiguities are in DATASET_AUDIT.md §1 and §13 A3. |
| Required citation | Dal Pozzolo, Caelen, Johnson & Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015 |
| Scope | **The complete 284,807-row file is used. No subsetting, no row deletion.** Every row is assigned to exactly one of train, validation, or test. |

**Why this dataset:**
- The transactions are real.
- It is an established benchmark.
- All features are continuous, which suits a Gaussian VAE.
- Legitimate transactions contain real dependency structure.
- Fraud is strongly separable in feature space.

The newer Mendeley candidate was rejected because its fraud labels can be inferred from artefacts and its features are independent of each other (DATASET_COMPARISON.md §9). The project owner chose this dataset over the other newer options considered (IEEE-CIS, BAF, TabFormer).

**Known limitations, to be disclosed in the report:**
- The data is from 2013 and spans only 2 days.
- The features are anonymised by PCA.
- The PCA was fitted by the data providers on all rows.
- There are about 70 frauds per evaluation split.

---

## 2. Proposal summary and agreed deviations

The proposal (IT429, Units 1–2) aims to build a VAE from scratch in PyTorch, train it on normal transactions only, and flag fraud by high reconstruction error, with Precision/Recall evaluation in a documented notebook on Colab.

| # | Proposal says | Final plan |
|---|---|---|
| G1 | Anomaly detection only | **Synthetic feature-vector generation is added as a core deliverable** (scope extension; guide to be informed, §13 A2) |
| G2 | "Unsupervised" | The primary experiment is described as **one-class / semi-supervised anomaly detection**. The fully unsupervised setting is the secondary experiment (§3.1). |
| G3 | "Learns the exact probability distribution" | The VAE maximises a lower bound (the ELBO) on the log-likelihood. It learns an **approximation**. |
| G4 | "Generative outperforms discriminative" | Not claimed. It is presented as a trade-off (the method needs no fraud labels for training). No supervised comparison is planned. |
| G5 | Features "amount, time, location, frequency" | The dataset has `Time`, 28 anonymised PCA components, and `Amount` only. Corrected in the report. |
| G6 | Single notebook deliverable | Code lives in `src/` and results in phase notebooks. Whether a single consolidated notebook is also needed is §13 A1. |

---

## 3. Final methodology

### 3.1 Terminology (used consistently in the report)

- **Unsupervised learning:** no labels are used anywhere in training, model selection, or threshold calibration. → **Experiment E2.**
- **One-class learning / novelty detection:** the model is trained on samples from the normal class only. → **Experiment E1.**
- **Semi-supervised anomaly detection** (in the anomaly-detection sense; Chandola, Banerjee & Kumar, 2009): labelled *normal* data is available for training and anomalies are not. → **Experiment E1.**

**Wording for E1:** *"A VAE that receives no labels as input, trained on legitimate transactions selected using ground-truth labels: a one-class (semi-supervised) anomaly detection protocol. Labels are used only to partition the data and to evaluate results after scoring."* E1 is **never** described as completely unsupervised.

### 3.2 Where labels are used

| Stage | E1 (primary) | E2 (secondary) |
|---|---|---|
| Model inputs (encoder and decoder) | **Never** | **Never** |
| Stratifying the train/val/test split | Yes (partitioning) | Yes (same shared split) |
| Choosing training rows | Yes: legitimate rows only (partitioning) | **No**: all training rows |
| Fitting the scaler | Legitimate training rows | All training rows |
| Early stopping | Validation loss on legitimate validation rows | Validation loss on all validation rows |
| Threshold calibration | Legitimate validation rows only (no fraud labels) | All validation rows (no labels) |
| Hyperparameters | Fixed in advance (§4.2); no label-based tuning | Same fixed values |
| Final evaluation | Yes, test labels, read **once**, after predictions are saved | Same |

Validation **fraud** rows are never used for any development decision: not for tuning, threshold selection, or model choice.

### 3.3 Data pipeline

1. **Load** `creditcard.csv` read-only and check its SHA-256. Cast `Class` to int.
2. **Assign groups.** Each distinct `V1–V28 + Amount` vector is a group (275,663 groups), so all copies of a vector share an ID. **No rows are dropped.**
3. **Stratified group split, 70 / 15 / 15** (train / validation / test), seed 42. Groups are shuffled within each class and allocated to splits until the row-count targets are reached. No duplicate group carries conflicting labels, so stratifying by group class is well defined. Expected sizes (exact numbers will be logged by Phase 3):

   | Split | Legitimate | Fraud |
   |---|---|---|
   | Train | ≈ 199,000 | ≈ 344 |
   | Validation | ≈ 42,650 | ≈ 74 |
   | Test | ≈ 42,650 | ≈ 74 |

   Split indices are saved to `data/processed/splits.json` and reused by every experiment. E1 and E2 share identical validation and test sets.
4. **Features (29):** `V1`–`V28` plus `log1p(Amount)`. **`Time` and `Class` are excluded** from model inputs.
5. **Scaling:** `StandardScaler` fitted only on the experiment's training rows (§3.2). Validation and test use that same fitted scaler and are never refitted.
6. **DataLoader:** returns feature tensors only. Labels sit in a separate array that the training code never imports.

### 3.4 Experiments

| ID | Role | Training data | Description |
|---|---|---|---|
| **E1** | **Primary** | Legitimate training rows (≈ 199k) | One-class VAE. Used for both anomaly detection and synthetic generation. |
| **E2** | **Secondary** | All training rows, unlabelled (≈ 0.17% fraud contamination) | Fully unsupervised VAE, same architecture and hyperparameters. Measures the effect of contamination. |

The hypothesis that E1 beats E2 is tested, not assumed. With about 0.17% contamination, the difference may be small or absent, and we will report whichever outcome occurs.

### 3.5 Threshold calibration

- τ is the **99th percentile** of anomaly scores on the experiment's validation rows (§3.2). This targets a false-positive rate of about 1% on legitimate traffic.
- The rule is fixed in advance, and **no label-based threshold optimisation is used**.
- The test set is evaluated at this τ. Behaviour at other thresholds is reported as a post-hoc sensitivity analysis on the test set (§6.2). It is never used to choose τ.
- **Order of operations on the test set:**
  1. Compute scores.
  2. Apply τ.
  3. Write predictions to disk.
  4. Load test labels.
  5. Compute metrics.

  Code assertions enforce this order.

---

## 4. Model: formulation and final architecture

### 4.1 Mathematical formulation

- **Prior:** p(z) = 𝒩(0, I).
- **Decoder:** p_θ(x | z) = 𝒩(x; f_θ(z), I). This is a Gaussian with fixed unit variance on standardised features.
- **Encoder:** q_φ(z | x) = 𝒩(z; μ_φ(x), diag(exp(log σ²_φ(x)))).

**Evidence lower bound:**

  ELBO(x) = 𝔼_{q_φ(z|x)}[ log p_θ(x | z) ] − KL( q_φ(z|x) ‖ p(z) ) ≤ log p_θ(x)

**Loss** (per sample, minimised; averaged over the batch):

  ℒ(x) = ½ ‖x − x̂‖² + β_t · KL,  with KL = −½ Σⱼ (1 + log σ²ⱼ − μⱼ² − σ²ⱼ)

This is the negative ELBO up to an additive constant. It matches the proposal's "MSE + KL" objective, with the squared error summed over features.

**Reparameterization:** z = μ + exp(½ log σ²) ⊙ ε, with ε ~ 𝒩(0, I).

### 4.2 Final architecture and hyperparameters (fixed in advance)

```
Encoder:  29 → Linear(64) → LeakyReLU → Linear(32) → LeakyReLU → { Linear(8) = μ ,  Linear(8) = log σ² }
Latent:   d = 8
Decoder:  8 → Linear(32) → LeakyReLU → Linear(64) → LeakyReLU → Linear(29)   (linear output)
Params:   ≈ 8.8k
```

| Setting | Value | Reason |
|---|---|---|
| Activation | LeakyReLU (slope 0.01) | Avoids dead units with heavy-tailed inputs |
| Normalisation | None (no BatchNorm) | Keeps generation independent of batch statistics |
| Dropout | None | The bottleneck is the regulariser; this keeps the model simple |
| β schedule | Linear warm-up from 0 to 1 over epochs 1–10, then β = 1 | Reduces posterior collapse |
| Optimiser | Adam, lr 1e-3, no weight decay | Standard, stable choice |
| Batch size | 512 | Fast on CPU, MPS, or Colab |
| Epochs | Up to **300** (planned 100; see D20); early stopping with patience 10, only after warm-up ends | Label-free stopping rule |
| Gradient clipping | Global norm 5.0 | Heavy-tailed features (kurtosis up to 933) |
| log σ² clamp | [−10, 10] | Numerical stability |

**Design rationale:**
- **Small bottleneck (8 dimensions for 29 inputs):** stops the network from learning the identity function, which would reconstruct anomalies well and defeat detection.
- **Linear decoder output:** standardised features are unbounded, so a bounded output layer would be wrong.
- **Implementation:** written entirely from scratch in `torch.nn`, with no pretrained models and no VAE libraries.

### 4.3 Anomaly scores

| Score | Definition | Role |
|---|---|---|
| **S1: reconstruction error** | ‖x − f_θ(μ_φ(x))‖² (deterministic, uses z = μ) | **Primary score** (as in the proposal) |
| S2: negative ELBO | ½‖x − x̂‖² + KL, averaged over L = 10 samples of z | Secondary |
| S3: importance-weighted estimate | −log (1/K) Σₖ p(x\|zₖ)p(zₖ)/q(zₖ\|x), K = 100 | Secondary: the closest available estimate of −log p_θ(x) |

S1 is a **distance**, not a probability. S2 and S3 estimate −log p_θ(x) **under the model**. None of them is P(fraud | x), and the report and app will say so.

---

## 5. Synthetic generation (core GenAI deliverable)

1. **Sampler:** z ~ 𝒩(0, I) → x̂ = f_θ(z), the decoder mean. Then undo the scaling, and turn `log1p(Amount)` back into `Amount = max(expm1(·), 0)` rounded to 2 decimals. Optional variants, reported separately:
   - temperature sampling with z ~ 𝒩(0, τ²I), τ ∈ {0.5, 0.8};
   - adding decoder noise;
   - latent interpolation between two real transactions.
2. **Source model:** E1, trained on legitimate transactions, so samples represent **legitimate-like** behaviour. E2 samples may be shown for comparison.
3. **Export:**
   - `data/synthetic/synthetic_e1_seed<k>_n<N>.csv` with columns `V1…V28, Amount`. There is **no `Class` column** and **no `Time` column**.
   - A metadata JSON alongside it: checkpoint SHA-256, seed, N, temperature, scaler file, creation date.
   - A `data/synthetic/README.md` carrying the ODbL attribution notice.
   - Default N = number of legitimate rows in the test split, so fidelity comparisons have equal sample sizes.
4. **Interpretability disclaimer** (in the README, the report, and the Streamlit app):

   > Generated samples are vectors in the **anonymised PCA feature space** of the source dataset (`V1–V28`) plus `Amount`. They are **not** interpretable real-world transaction records. The original features and the PCA loadings were never published, so samples cannot be mapped back to merchants, cards, locations, or customers. `Amount` is the only directly interpretable field. Sample quality can be judged only **statistically**, by agreement of distributions and dependencies with real data, not semantically.

5. **Memorisation check:** compare each synthetic sample's distance to its closest real training record with the same distance for real held-out records. If synthetic samples sit much closer to training data than held-out data does, the model is copying training rows.

---

## 6. Evaluation (final)

### 6.1 Generative quality (E1 samples vs **test-split legitimate rows**, which no development decision touched)

| Check | Metric |
|---|---|
| Marginals | Histogram and KDE overlays; per-feature Kolmogorov–Smirnov statistic and Wasserstein-1 distance |
| Tails | Quantiles at 0.1/1/5/95/99/99.9% compared |
| Dependencies | Correlation heatmaps (real, synthetic, difference) and the Frobenius norm of the difference, focusing on the **Amount–V2/V5/V7/V20** correlations |
| Joint structure | PCA (fitted on real data) 2-D overlay; optional t-SNE on a 5k subsample |
| Detectability | Classifier two-sample test: logistic regression and histogram gradient boosting, 5-fold CV ROC-AUC. 0.5 means indistinguishable. |
| Validity | % finite, % `Amount ≥ 0`, % within the observed range |
| Latent space | μ scatter (a separate d = 2 visualisation model, plus PCA of μ for d = 8), coloured by class **after** training; per-dimension KL (active units) |
| Reconstruction | Per-feature reconstruction MSE on legitimate rows; real vs reconstructed plots |
| Memorisation | Distance-to-closest-record comparison (§5.5) |

### 6.2 Anomaly detection (test split, read once)

| Metric | Notes |
|---|---|
| **Average Precision (PR-AUC)** | **Primary metric.** A random scorer gets about 0.0017. |
| Precision, Recall, F1 at τ | τ from §3.5 |
| Confusion matrix at τ | Counts and rates |
| Precision–Recall curve | With the τ operating point marked |
| ROC-AUC | Secondary; flagged as optimistic under class imbalance |
| Recall at fixed alert budgets | Top 100 / 500 / 1,000 scores |
| Threshold sensitivity (post-hoc) | Precision/Recall/F1 when τ is set at validation percentiles q ∈ {90, 95, 97.5, 99, 99.5, 99.9} |
| Error analysis | Profiles of false negatives vs true positives; the highest-scoring false positives |
| **E1 vs E2** | Side-by-side on the same test set; paired bootstrap of the AP difference |

**Statistics:**
- **Seeds:** 5 seeds (0–4) per configuration, reported as mean ± std.
- **Confidence intervals:** 95% bootstrap intervals from 1,000 class-stratified resamples of the test set.

### 6.3 Sensitivity analyses (fixed list; reported as analyses, never used to select the headline model)

- Latent dimension d ∈ {2, 4, 16}.
- β ∈ {0.5, 2}.
- Deterministic autoencoder baseline (same network, no KL, no sampling).
- Score S1 vs S2 vs S3.
- `RobustScaler`.
- Adding sin/cos hour-of-cycle features.
- Time-ordered split.
- Exact-duplicate-removed data.

The **headline result is always the fixed E1 configuration in §4.2.**

---

## 7. Reproducibility requirements

1. **Data integrity:** check the SHA-256 at every load and fail on mismatch. The raw file is opened read-only and never written.
2. **Seeds:** `random`, `numpy`, and `torch` are seeded per run. DataLoader uses a seeded `generator` and `worker_init_fn`. The split seed (42) is fixed separately from the model seeds (0–4).
3. **Determinism:** headline runs execute on **CPU** with `torch.use_deterministic_algorithms(True)`, because MPS and CUDA do not guarantee bitwise reproducibility. MPS or Colab may be used for development, and the device is logged for every run.
4. **Configuration:** every run is driven by `configs/*.yaml`. The resolved config is saved with the run.
5. **Artefacts per run** (`outputs/runs/<run_id>/`): config, `pip freeze`, git commit (if a repo exists), device, training log CSV, checkpoint, scaler, score arrays, predictions, metrics JSON.
6. **Saved splits:** `data/processed/splits.json` plus a split summary. Automated checks:
   - no row appears in two splits;
   - no group appears in two splits;
   - all 284,807 rows are assigned;
   - class rates are stable across splits.
7. **Environment:** pinned `requirements.txt` (versions above), with `pyarrow` excluded unless needed.
8. **Traceability:** every number in the report traces back to a metrics file and a run ID.

---

## 8. Final directory structure

The existing files stay where they are. **The dataset is not moved or copied**; its path is set in the config.

```
genAI/
├── creditcard.csv                     # EXISTING, immutable source of truth
├── Project Proposal_ ... .pdf          # EXISTING
├── PROJECT_PLAN.md · DATASET_AUDIT.md · DATASET_COMPARISON.md
├── README.md                          # setup, reproduction, attribution (ODbL), disclaimer
├── requirements.txt
├── .gitignore                         # creditcard.csv, outputs/checkpoints, data/processed caches
├── configs/
│   ├── e1_primary.yaml
│   ├── e2_unsupervised.yaml
│   └── sensitivity/*.yaml
├── data/
│   ├── processed/                     # splits.json, split_summary.csv, transformer_e1.json, transformer_e2.json (portable JSON, not pickles)
│   └── synthetic/                     # generated CSVs + metadata JSON + README (disclaimer, licence)
├── src/
│   ├── utils/        {seed.py, device.py, io.py (hash check, run dirs)}
│   ├── data/         {load.py, split.py, preprocess.py, dataset.py}
│   ├── models/       {vae.py, autoencoder.py}
│   ├── training/     {losses.py, train.py}
│   ├── generation/   {sample.py}
│   ├── detection/    {scores.py, threshold.py}
│   └── evaluation/   {detection_metrics.py, generative_metrics.py, bootstrap.py, plots.py}
├── scripts/                           # CLI entry points (run_eda.py, prepare_data.py) + run_notebook.py (executes notebooks without ipykernel)
├── tests/                             # unit tests: KL, reparameterization, shapes, split integrity, label isolation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_splits.ipynb
│   ├── 03_vae_training.ipynb          # Colab-compatible
│   ├── 04_synthetic_generation.ipynb
│   ├── 05_anomaly_detection.ipynb
│   └── 06_experiments_comparison.ipynb
├── outputs/
│   ├── runs/<run_id>/                 # per-run artefacts (§7.5)
│   ├── figures/
│   └── metrics/                       # aggregated summary tables
├── app/
│   └── streamlit_app.py
└── reports/
```

---

## 9. Phased roadmap

| Phase | Objective | Main files | Verification | Depends on |
|---|---|---|---|---|
| **1. Investigation** ✅ | Audit the data and finalise the plan | the planning `.md` files | Owner approves §13 | — |
| **2. EDA** ✅ | Figure-backed EDA | `01_eda.ipynb`, `src/data/load.py`, `plots.py` | Numbers match DATASET_AUDIT.md; hash check passes | 1 |
| **3. Preprocessing & splits** ✅ | Deterministic stratified group split; scalers | `src/data/*`, `configs/`, `02_*.ipynb` | Automated split checks (§7.6); label absent from feature tensors; re-run gives identical indices | 2 |
| **4. Formulation** | Written derivation (ELBO, KL, reparameterization, scores) | `reports/vae_formulation.md` | Matches §4.1 | 1 (parallel with 2–3) |
| **5. Implementation** ✅ | From-scratch VAE + AE baseline | `src/models/*`, `losses.py`, `tests/` | KL matches a Monte-Carlo estimate (< 1%); KL = 0 at μ = 0, log σ² = 0; gradients flow through the reparameterization; a 256-row overfit sanity check succeeds | 3, 4 |
| **6. Training** ✅ (E1, E2 × 5 seeds; architecture sensitivity runs pending) | E1 and E2 × 5 seeds, plus sensitivity runs | `train.py`, `03_*.ipynb` | Losses converge without NaNs; ≥ 2 active latent units; CPU runs reproducible | 5 |
| **7. Generation** ✅ | Sample, export, evaluate fidelity | `sample.py`, `generative_metrics.py`, `04_*.ipynb` | 100% valid rows; all §6.1 metrics reported; memorisation check; disclaimer present | 6 |
| **8. Detection** ✅ | Scores, τ, single test evaluation | `scores.py`, `threshold.py`, `05_*.ipynb` | Order of operations enforced (§3.5); τ from validation only; bootstrap CIs | 6 |
| **9. Comparison** ✅ (E1 vs E2; architecture sensitivity pending) | E1 vs E2, sensitivity analyses, error analysis | `06_*.ipynb` | Every number traceable to a run ID; headline = the fixed configuration | 7, 8 |
| **10. App** | Streamlit demo: generate + download, score rows, latent plot, metrics | `app/streamlit_app.py` | Runs locally; scores match Phase 8; shows the score-is-not-a-probability and PCA-space disclaimers | 7–9 |
| **11. Documentation** ✅ (final report, Colab notebook + execution report; slides pending) | README, report, slides, (consolidated notebook) | `README.md`, `reports/` | A fresh-environment run reproduces the headline numbers within seed variance; terminology follows §3.1; attribution included | all |

---

## 10. Technical risks

| Risk | Mitigation |
|---|---|
| Heavy tails dominate the MSE loss | Gradient clipping; `RobustScaler` sensitivity run; monitor per-feature loss |
| VAE reconstructs anomalies too well | Small bottleneck; d and AE sensitivity runs |
| Posterior collapse | KL warm-up; per-dimension KL monitoring |
| Synthetic samples blurred, tails too thin | Report honestly; temperature analysis |
| Few test frauds (≈ 74) | Bootstrap CIs, 5 seeds, cautious interpretation |
| Leakage (scaler, duplicates, threshold, labels) | Automated assertions in Phases 3 and 8 |
| Overclaiming (exact distribution, fraud probability, unsupervised, realistic transactions) | Terminology rules in §2, §3.1, §4.3, §5.4 |
| Licence non-compliance when publishing | Attribution everywhere; ODbL for any public derived data (§13 A3) |

---

## 11. Decision register (finalised)

| # | Decision | Final choice | Rationale |
|---|---|---|---|
| D1 | Dataset | `creditcard.csv`, all 284,807 rows | Chosen by the owner; real data with an established benchmark |
| D2 | Dataset location | Stays at the project root, referenced from config; not moved or copied | No risk to the source file; saves 144 MB |
| D3 | Duplicates | Keep every row; stratified **group** split | Satisfies "no subset" and still prevents cross-split leakage; dedup only as a sensitivity run |
| D4 | Split | 70/15/15 stratified group-random, seed 42; time-ordered split as a sensitivity run | Maximises test frauds; comparable with published work |
| D5 | Features | V1–V28 + `log1p(Amount)`; raw `Time` excluded; hour sin/cos as a sensitivity run | `Time` acts as a row index; `Amount` skew 17 → 0.16 after log |
| D6 | Scaling | `StandardScaler` fitted on the experiment's training rows only | Leakage-safe; `RobustScaler` as a sensitivity run |
| D7 | Primary experiment | E1, one-class VAE on legitimate training rows | As proposed |
| D8 | Secondary experiment | E2, fully unsupervised VAE on all training rows | Measures contamination effect |
| D9 | Threshold | 99th percentile of validation scores (≈ 1% FPR); no label-based tuning | Consistent with labels being used only for partitioning |
| D10 | Model selection | Hyperparameters fixed in advance; early stopping on validation loss only | No label-driven tuning; prevents test overfitting |
| D11 | Architecture | 29-64-32-8 MLP VAE as in §4.2 | Small, fast, fits the data |
| D12 | Primary score | S1 reconstruction error; S2 and S3 as secondary | Matches the proposal; likelihood-style scores kept separate |
| D13 | Synthetic scope | Legitimate-like vectors from E1; no labels attached; PCA-space disclaimer | Core GenAI deliverable; honest about interpretability |
| D14 | Baseline | Deterministic AE (same network) as a sensitivity run; no supervised classifiers | Isolates the contribution of the variational part; keeps the VAE primary |
| D15 | Primary metric | Average Precision (PR-AUC) | Recommended by the dataset owners; right for 0.17% prevalence |
| D16 | Statistics | 5 seeds; 1,000-resample stratified bootstrap 95% CIs | ≈ 74 test frauds |
| D17 | Compute | Local CPU/MPS primary (small model); Colab optional; headline runs on CPU | Reproducibility; no dependence on Colab |
| D18 | Generative fidelity reference | Test-split legitimate rows | Not touched by any development decision |
| D19 | Final deliverable | Self-contained Colab notebook + `src/` + phase notebooks | Owner decision A1; matches the proposal |
| D20 | Max epochs (deviation, 2026-11-07) | Raised from 100 to 300 | E1 seed-0 validation loss was still falling at epoch 100; decided from validation loss only, before any test evaluation. All runs early-stopped (best epochs 103–246). |
| D21 | Generation variant (2026-11-07) | The pre-listed "decoder noise" variant is reported alongside the primary decoder-mean sampler, with noise std fitted on training residuals | Decoder-mean samples showed about half the real spread. Both samplers are reported; neither is selected using test data. |

---

## 12. Open questions (non-blocking)

- Whether this copy is byte-identical to Kaggle's file (size matches; a checksum needs a Kaggle login).
- The real-world time of day for `Time` = 0, and the currency of `Amount` (presumed EUR, unverified).
- **No experiments have been run. Nothing in this plan is a result.**

---

## 13. Owner decisions (resolved 2026-11-07)

| # | Decision | Resolution |
|---|---|---|
| **A1** | Submission format | **A single self-contained Google Colab notebook is a required deliverable**, alongside `src/` and the phase notebooks. It must run end to end on a fresh Colab runtime: upload or mount `creditcard.csv`, check the hash, then split, train, generate, detect, and evaluate. |
| **A2** | Guide sign-off and terminology | **Settled.** The project owner confirms the guide accepts the scope extension. The current title is kept, and the report adds one sentence stating that the primary protocol is one-class (semi-supervised) and that E2 is the fully unsupervised comparison. |
| **A3** | Public release | **Not a concern for now** (class project). If the repo is later made public on GitHub, two rules apply: do not commit `creditcard.csv`, and keep the dataset attribution line in the README. |
| **A4** | Machine setup | **Git is handled by the project owner.** Claude does not run git commands. No packages are installed. Notebooks run on Colab, and local work uses the existing Python environment. `ipykernel` is only needed if the team later wants to run notebooks locally in VS Code. |

**All blocking decisions are resolved. The plan is ready for implementation, starting with Phase 2.**
