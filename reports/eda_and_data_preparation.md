# EDA and Data Preparation: Summary

These are roadmap Phases 2 and 3 in PROJECT_PLAN.md (the project brief called them Phase 1 and Phase 2). Everything below was produced by code in this repository:
- **Notebooks:** [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) and [notebooks/02_preprocessing_and_splits.ipynb](../notebooks/02_preprocessing_and_splits.ipynb)
- **Scripts:** `scripts/run_eda.py` and `scripts/prepare_data.py`

## Dataset statistics

| Item | Value |
|---|---|
| Rows × columns | 284,807 × 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Normal / fraud | 284,315 / 492 |
| Fraud percentage | **0.1727%** (1 fraud per 578 normal) |
| Missing / infinite / non-numeric | 0 / 0 / none |
| Exact duplicate rows | 1,081 (1,062 normal, 19 fraud) |
| Rows repeating a feature vector (ignoring `Time`) | 9,144; 275,663 distinct vectors; none with conflicting labels |
| `Amount` | 0–25,691.16; median 22.00; skew 16.98, which falls to 0.16 after `log1p`; 1,825 zeros |
| `Time` | 0–172,792 s (48 h), sorted; clear day/night volume cycle |
| V1–V28 | centred; std 1.96 → 0.33; excess kurtosis up to 933; uncorrelated over all rows |

The full statistics are in `outputs/metrics/eda_stats.json`, `eda_feature_summary.csv` and `eda_class_comparison.csv`.

## Figures (`outputs/figures/eda/`)

| File | Shows |
|---|---|
| `01_class_counts.png` | Normal vs fraud counts (linear and log scale) |
| `02_class_distribution.png` | Class shares, with a zoom on the last 1% |
| `03_amount_distribution.png` | Raw Amount, log1p(Amount) by class, and cumulative distribution by class |
| `04_time_distribution.png` | Normal and fraud counts per hour, and fraud rate per hour |
| `05_feature_scales.png` | Std and excess kurtosis of V1–V28 |
| `06_feature_distributions_all.png` | All 28 V features by class |
| `07_selected_features.png` | Box plots of the six most separated features |
| `08_correlation_heatmaps.png` | Correlations over all rows, normal rows and fraud rows |
| `09_normal_vs_fraud_comparison.png` | Standardised mean difference per feature |

## Key findings

1. **Imbalance.** The imbalance is extreme (0.1727%), so PR-AUC is the primary metric and evaluation sets keep the natural fraud rate.
2. **Data quality.** The data has no missing or invalid values. Duplicates exist, but no feature vector carries conflicting labels. Duplicates are kept and handled by a group-aware split.
3. **`Amount`.** Heavily skewed, so it is modelled as `log1p(Amount)`. Frauds have a lower median amount (9.25 vs 22.00).
4. **`Time`.** Shows a daily cycle, with the highest fraud rates in the low-volume hours (1.71% in hour 2). The clock time of hour 0 is undocumented. Raw `Time` is excluded from model inputs.
5. **V1–V28.** Centred but not unit-scaled, and heavy-tailed, so all inputs are standardised and extreme values are kept.
6. **Correlations.** The features are orthogonal over all rows but correlated within each class (up to |r| = 0.17 for normal rows and 0.97 for fraud rows). `Amount` correlates with V2 (0.53), V7 (0.40) and V5 (0.39).
7. **Class separation (descriptive only).** Fraud means sit far from normal means on V17 (−8.9 std), V14 (−7.8 std), V12 (−6.6 std) and V10 (−5.4 std). 82.7% of frauds have a feature beyond |z| = 5, against 2.7% of normal rows. No claim is made about what any PCA component represents.

## Partitions (stratified group split, seed 42)

| Split | Rows | Normal | Fraud | Fraud % |
|---|---|---|---|---|
| Train | 199,366 | 199,021 | 345 | 0.1730 |
| Validation | 42,721 | 42,647 | 74 | 0.1732 |
| Test | 42,720 | 42,647 | 73 | 0.1709 |
| **Total** | **284,807** | **284,315** | **492** | 0.1727 |

| Partition | Rows | Used for |
|---|---|---|
| E1 train | 199,021 normal | Primary VAE training |
| E1 calibration | 42,647 normal | Early stopping and anomaly threshold (E1) |
| E2 train / E2 validation | 199,366 / 42,721, unlabelled | Unsupervised VAE (secondary) |
| Test | 42,647 normal + 73 fraud | Final evaluation only |
| Train frauds (345) | — | Excluded from E1; included in E2 without labels |
| Validation frauds (74) | — | Not used for any decision |

## Preprocessing

- **Model inputs:** `V1`–`V28` and `log1p(Amount)` (29 features). `Class` and `Time` are excluded.
- **Scaling:** standardisation using mean and population std, fitted on E1 training rows (normal only) for E1 and on all training rows for E2. Validation and test are only transformed with these statistics.
- **Saved transformers:** `data/processed/transformer_e1.json` and `transformer_e2.json`. They are plain JSON, so they don't depend on a scikit-learn version. They provide `inverse_transform` (un-standardise, `expm1`, clip `Amount` ≥ 0, round to 2 decimals) for exporting synthetic data.
- **Saved split:** `data/processed/splits.json`, with a SHA-256 fingerprint per index list that is verified on load.

## Leakage checks: 18 of 18 passed (`outputs/metrics/leakage_checks.json`)

- **Partition integrity:** the splits are disjoint and cover all 284,807 rows.
- **No duplicate leakage:** no duplicate group spans two splits, and an independent hash check confirms no identical vector appears in both train and validation or train and test.
- **Stratification:** the fraud rate is preserved in each split (±0.02 percentage points).
- **Partition roles:** E1 training and calibration sets contain no fraud; the test set contains both classes.
- **Label isolation:** model inputs exclude `Class` and `Time`.
- **Scalers:** each scaler's statistics equal those recomputed from its training rows only and differ from validation/test statistics. E1 training data is exactly standardised.
- **Round trip:** the inverse transform reproduces the original values (error below 1e-11).
- **Reproducibility:** the split is identical when re-run with the same seed.
- **Raw data:** the dataset's SHA-256 is unchanged.

The tests in `tests/test_data_pipeline.py` pass 7 of 7.
