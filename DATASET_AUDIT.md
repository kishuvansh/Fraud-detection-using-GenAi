# Dataset Audit — `creditcard.csv`

Audit date: 2026-11-06 · **Status: selected as the sole primary project dataset** (see PROJECT_PLAN.md §1). The full 284,807-row file is used; no subset is taken.
Method: read-only inspection with pandas 2.3.1 / NumPy 2.3.2 (Python 3.13.5). The original file was not modified, copied, or moved. Every number below was computed from the file during this audit.

---

## 1. File identity

| Item | Value |
|---|---|
| Filename | `creditcard.csv` |
| Location | project root: `/Users/vansh/Downloads/genAI/creditcard.csv` |
| Format | Comma-separated text, header row, values unquoted except `Class` (stored as `"0"` / `"1"`) |
| Size | 150,828,752 bytes (~144 MB) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| In-memory size (pandas, float64) | ~70.6 MB |

Record the SHA-256 above. Any later pipeline step should verify it before reading, so we know the source of truth has not changed.

### Is this the ULB / Kaggle Credit Card Fraud dataset?

The contents match it on every check we can make from the file itself:

- 284,807 rows, 31 columns: `Time`, `V1`…`V28`, `Amount`, `Class`.
- Exactly 492 positive labels (0.1727%).
- `Time` spans 0 to 172,792 seconds (about 48 hours).
- `V1`…`V28` have the signature of PCA output: mean ≈ 0, standard deviations strictly decreasing from V1 to V28, and pairwise correlations ≈ 0 on the full data (details in §5).

### Source, attribution and licence (checked 2026-11-06 against Kaggle's public metadata API)

| Item | Value | Status |
|---|---|---|
| Official source | Kaggle `mlg-ulb/creditcardfraud`, "Credit Card Fraud Detection" (Kaggle dataset id 310) | Verified |
| Owner | Machine Learning Group, Université Libre de Bruxelles (ULB). The uploader shown is "Timo Bozsolik"; version 2 is credited to "Andrea". | Verified |
| Version | v3 (notes: "Fixed preview"), last updated 2018-03-23. v2 introduced the CSV format. | Verified |
| Kaggle file size | 150,828,752 bytes, **exactly equal** to the local file | Verified (size match) |
| Content description | Card transactions by European cardholders, September 2013, two days, 492 frauds out of 284,807. V1–V28 are PCA components; `Time` and `Amount` are not transformed. Kaggle recommends AUPRC as the metric. | Verified, and consistent with this audit |
| Data origin | Research collaboration between **Worldline** and ULB MLG | Stated by the owner; cannot be verified independently |
| Licence (Kaggle) | **"Database: Open Database, Contents: Database Contents"**, i.e. **ODbL 1.0** for the database and **DbCL 1.0** for its contents | Verified as listed |
| Required citation | Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM). Kaggle lists further optional works by the same group. | Verified |
| Kaggle popularity | About 1.2 million downloads, 6,320 public notebooks | Verified |
| Mirror | OpenML dataset 1597 (uploaded 2015-06-25). It credits Dal Pozzolo, Caelen, and Bontempi and lists its licence as "Public". | Verified |

**Unresolved licensing ambiguities:**
1. **The sources disagree on the licence.** Kaggle lists ODbL/DbCL, while OpenML lists "Public". We will treat the Kaggle licence, which is the stricter of the two, as binding.
2. **ODbL share-alike.** Under ODbL, a *Derivative Database* that is used publicly must itself be offered under ODbL, and a *Produced Work* must carry an attribution notice. It is not settled whether VAE-generated synthetic rows or our processed split files count as a "Derivative Database" or a "Produced Work". The safe course is to attribute the source on everything we publish, and to release any publicly shared synthetic CSV or processed data under ODbL 1.0. This is our reading of the licence, not legal advice.
3. **Upstream rights.** No licence statement from Worldline itself was found. We rely on ULB MLG's publication on Kaggle.
4. **Byte-level identity.** The checksum of Kaggle's file cannot be retrieved without signing in. Identity rests on the exact byte-size match plus the content checks above.

Required attribution text for the report and README: *"Credit Card Fraud Detection dataset, Machine Learning Group – ULB / Worldline, Kaggle (mlg-ulb/creditcardfraud), ODbL 1.0 / DbCL 1.0; Dal Pozzolo et al., CIDM 2015."*

---

## 2. Shape, columns, and types

- **Rows:** 284,807
- **Columns:** 31

| Column(s) | Parsed dtype | Meaning (from content and documentation) |
|---|---|---|
| `Time` | float64 (whole numbers only) | Seconds elapsed since the first transaction in the file. Not a timestamp or time of day. |
| `V1` … `V28` | float64 | Anonymised principal components. Original features and PCA loadings are not available. |
| `Amount` | float64 (at most 2 decimals) | Transaction amount. Raw scale, not PCA-transformed. Currency not stated in the file. |
| `Class` | int64 after parsing (stored as quoted strings) | Label: 0 = legitimate, 1 = fraud |

Note: `Class` is written as `"0"`/`"1"` with quotes. pandas parses it to int64 by default, but loaders should cast it explicitly (`astype(int)`) rather than rely on that.

---

## 3. Data quality checks

| Check | Result |
|---|---|
| Missing values (all columns) | **0** |
| Infinite values | **0** |
| Exact duplicate rows (all 31 columns) | **1,081** (1,062 legitimate, 19 fraud) |
| Duplicate rows ignoring `Time` | **9,144** rows repeat an earlier `V1–V28 + Amount + Class` vector |
| Distinct `V1–V28 + Amount` vectors | 275,663 |
| Vectors that occur more than once | 5,149 (most often 2 copies; the largest group has 77 copies) |
| Duplicate vectors with conflicting labels | **0** (no feature vector appears with both labels) |
| Fraud rows / distinct fraud vectors | 492 / 473 |
| Constant or near-constant columns | **None.** Every V column has 275,663 unique values; the most frequent value of any V column covers 0.027% of rows. `Amount` has 32,767 unique values; its most frequent value covers 4.81% of rows. |
| `Amount == 0` | 1,825 rows (1,798 legitimate, 27 fraud) |
| `Time` sorted ascending | Yes (monotonic, 124,592 unique values) |

The largest duplicate group (77 copies) is a legitimate transaction of 8.99 repeated between t = 26,310 s and t = 68,045 s. It looks like a recurring charge, not a file error.

**Interpretation.** "No missing values" does **not** mean the data is ready for modelling:

1. **Exact duplicates (1,081 rows)** may be logging artefacts or genuine repeat charges recorded in the same second. We cannot tell which.
2. **Near-duplicates (same features, different `Time`)** are plausibly genuine repeat transactions. If identical feature vectors land in different splits, a reconstruction model has effectively seen those test points already. With a random 70/15/15 split, 2,763 duplicate groups would cross a split boundary. With a time-ordered split, 1,335 would, and none of them are fraud.

**Decision (final):** **all 284,807 rows are kept.** Duplicates are handled by a **group-aware split**: every copy of an identical `V1–V28 + Amount` vector goes into the same split. That removes the cross-split leakage without deleting any data. A deduplicated re-run is included only as a sensitivity analysis (PROJECT_PLAN.md §6.3).
3. **Scaling is not complete.** `Amount` and `Time` are on their raw scales (§5).
4. **Heavy tails are present** (§5) and will affect both the loss function and synthetic sample quality.

---

## 4. Target column and class balance

`Class` is the target.

| Subset | Legitimate (0) | Fraud (1) | Fraud % |
|---|---|---|---|
| Full file | 284,315 | 492 | **0.1727%** |
| After dropping exact duplicates | 283,253 | 473 | 0.1667% |
| Day 1 (`Time` < 86,400) | 144,505 | 281 | 0.194% |
| Day 2 (`Time` ≥ 86,400) | 139,810 | 211 | 0.151% |

The imbalance is extreme (about 1 fraud per 579 legitimate transactions). This has three consequences:
- ROC-AUC will look high even for weak detectors. Precision–recall metrics are the appropriate primary measure.
- Any test split will hold only about 50–75 frauds. Metrics computed on so few positives have wide uncertainty, so we should report bootstrap confidence intervals.
- Precision depends on prevalence, so the test set must keep the natural fraud rate. It should not be rebalanced.

---

## 5. Feature distributions

### 5.1 `V1` … `V28` (PCA components)

| Property | Finding |
|---|---|
| Means (full data) | ≈ 0 (largest absolute mean 4.9 × 10⁻¹⁵) |
| Standard deviations | Strictly decreasing: V1 = 1.959 → V28 = 0.330. **Not** unit-scaled. |
| Pairwise correlation (full data) | ≤ 2.3 × 10⁻¹⁴, i.e. orthogonal |
| Pairwise correlation (legitimate rows only) | Up to 0.169 in absolute value |
| Skewness | −8.52 (V8) to +11.19 (V28) |
| Excess kurtosis | 0.20 (V13) to **933.4** (V28). V5, V7, V8, V20, V21, V23, V27, and V28 all exceed 200. |
| Range | V5 spans −113.7 to 34.8; V7 reaches 120.6. The 0.1%–99.9% range of legitimate rows is about −15.1 to 9.1. |
| Rows with any |z| > 5 | 7,952 total: 7,545 legitimate and **407 fraud (82.7% of all frauds)** |

**Conclusions:**
- The V features are centred and decorrelated, but they are **not standardised** (their variances differ by a factor of about 35) and **not Gaussian** (several are extremely heavy-tailed).
- The PCA was fitted by the data providers on the **entire** dataset, including rows that will become our validation and test sets. This is a small, unavoidable leak from the source. We cannot undo it, so we will state it in the report.
- Fraud rows are strongly shifted on a few components. Measured in legitimate-class standard deviations, the absolute mean shift is V17 8.91σ, V14 7.79σ, V12 6.63σ, V10 5.45σ, V16 4.91σ, V3 4.83σ. This is the signal a reconstruction-based detector can exploit. (These label-based statistics are for the audit only and must not drive feature selection; see §7.)

### 5.2 `Amount`

| Statistic | All | Legitimate | Fraud |
|---|---|---|---|
| Mean | 88.35 | 88.29 | 122.21 |
| Median | 22.00 | 22.00 | 9.25 |
| 75th percentile | 77.17 | 77.05 | 105.89 |
| Max | 25,691.16 | 25,691.16 | 2,125.87 |

- Skewness 16.98, excess kurtosis 845. Quantiles: 90% = 203.0, 99% = 1,017.97, 99.9% = 3,000.
- `log1p(Amount)` reduces skewness to 0.16, so a log transform followed by standardisation is a good fit.
- `Amount` is correlated with several components: V2 (|r| = 0.53), V7 (0.40), V5 (0.39), V20 (0.34). A generative model should reproduce these relationships, which gives us a useful fidelity check.

### 5.3 `Time`

- Covers 0–172,792 s, about two consecutive days, in ascending order.
- Taking `Time mod 86,400` gives the hour since the start of each 24-hour cycle. Volume drops sharply in hours 1–6 (e.g. 2,186 transactions in hour 4, against about 16,500 in hours 10–21). The fraud rate is highest in hour 2 (1.71%) and hour 4 (1.04%), against roughly 0.05–0.2% during busy hours.
- Whether hour 0 corresponds to midnight is **not known**. The low-volume period suggests night time, but that is an inference, not a fact.
- Raw `Time` correlates with V3 (|r| = 0.42) and V5 (0.17).

**Conclusion:** raw `Time` works like a row index, not a behavioural feature. In a time-ordered split, test values would lie outside the training range. We recommend excluding raw `Time` from the model inputs. A cyclical hour feature (sin/cos of `Time mod 86,400`) can be tested as an ablation.

### 5.4 Correlation with the label (audit only)

The strongest point-biserial correlations with `Class` (absolute values): V17 0.33, V14 0.30, V12 0.26, V10 0.22, V16 0.20, V3 0.19, V7 0.19.

---

## 6. Is preprocessing complete?

**No.** The file is clean in the narrow sense: no missing values, no infinities, valid types. It is not ready for modelling:

| Item | Status | Required action |
|---|---|---|
| Missing / infinite values | Complete | None |
| Exact duplicates | **Not handled** | Keep all rows; split by group so copies never cross splits. Deduplicated run as a sensitivity analysis only. |
| Near-duplicate vectors across splits | **Not handled** | Split by group (the same mechanism) |
| `Amount` scale and skew | **Not handled** | `log1p`, then standardise using statistics from the training split only |
| `Time` | **Not handled** | Drop raw `Time`; optional cyclical hour feature as an ablation |
| V-feature scale | Partially (centred only) | Standardise using statistics from the training split only |
| Heavy tails | Not handled | Keep the extreme values (they carry the anomaly signal). Handle robustness in the loss or scaler choice, not by deleting outliers. |
| Train/val/test split | Not present | Leakage-safe stratified group split (PROJECT_PLAN.md §3.3) |

---

## 7. Data leakage risks

| Risk | Severity | Mitigation |
|---|---|---|
| Scaler fitted on all data | High if done wrong | Fit scalers on training rows only (legitimate rows only for the primary model) |
| Duplicate or near-duplicate vectors in both train and test | Medium | Split by group (all copies in one split); include an automated assertion that no group ID appears in two splits |
| Using `Class` as a model input | Critical if done wrong | `Class` never enters the encoder or decoder; the loader returns it separately |
| Choosing the threshold on test data | High | Choose the threshold on the validation split only; evaluate the test set once |
| Choosing features or hyperparameters from label correlations (§5.1, §5.4) | Medium | Keep all 28 V features. Select hyperparameters on validation data only. |
| Provider PCA fitted on the full dataset | Low, unavoidable | Disclose it in the report |
| Raw `Time` as an input | Medium (temporal shortcut, out-of-range at test time) | Exclude it |
| Repeated test evaluation while tuning | High | Lock the test set; evaluate it once per final configuration |

---

## 8. Is the dataset suitable for a tabular VAE?

**Yes, with caveats.**

Strengths:
- All features are continuous, and there are no categorical columns to encode. This is the easiest case for a Gaussian-decoder VAE.
- 29–31 low-dimensional features, so a small MLP VAE fits easily in memory.
- About 199k legitimate training rows is plenty for a model of this size.
- Ground-truth labels are available for evaluation after scoring, and the fraud signal is concentrated in a few components.

Caveats:
- **Heavy tails.** A Gaussian decoder with MSE loss will be dominated by a small number of extreme rows during training. It will also tend to under-generate tails, so synthetic samples will look smoother than real data.
- **Anonymised features.** We cannot judge whether synthetic samples are *semantically* plausible (for example, "is this a realistic merchant category"). We can only check statistical fidelity. Validity checks are limited to `Amount ≥ 0`, 2-decimal rounding, and agreement of distributions and dependencies.
- **Few frauds** (about 50–75 per test split) means wide confidence intervals on precision and recall.
- **Short time window** (2 days), so we cannot evaluate long-horizon drift.
- **No features for location, merchant, or customer history.** The proposal mentions "location, frequency"; those features are not in this dataset.

---

## 9. Open questions

1. Whether this copy is byte-identical to the official Kaggle release. The byte size matches exactly, but Kaggle's checksum requires signing in.
2. Licensing ambiguities: see §1 (Kaggle ODbL vs OpenML "Public", and ODbL share-alike for synthetic or derived data).
3. The real-world meaning of `Time` = 0 (calendar time of day).
4. The currency of `Amount`. Kaggle does not state it; transactions by European cardholders suggest EUR, but that is an assumption.
5. Whether near-duplicate vectors are genuine repeat purchases or artefacts. We treat them as genuine but keep each group within one split.
