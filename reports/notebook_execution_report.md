# Notebook Execution Report: `colab/VAE_Financial_Transaction_Modeling_Colab.ipynb`

**Verification date:** 2026-11-07 · **Result: PASSED.** All 32 code cells executed top to bottom with 0 errors, and 10 of 10 built-in verification checks passed.

## How it was executed

- **Fresh process:** a new Python process ran `python scripts/run_notebook.py colab/VAE_Financial_Transaction_Modeling_Colab.ipynb`. It executes every cell in order in an empty namespace, so no variable from an earlier session or manual step was available. This is the local equivalent of *Restart runtime → Run all*.
- **Clean start:** `colab/vae_outputs/` was deleted beforehand, so every output was produced by this run.
- **Working directory:** `colab/`. The notebook found the dataset automatically at `../creditcard.csv` (one of its built-in search paths) and verified it.
- **Saved outputs:** the executed notebook was saved with all its outputs, including 20 inline figures.
- **Later edit:** after the run, one Markdown cell (section 15) gained a note on the instability of the linear two-sample test. No code cell or output changed.

## Environment and seeds

| Item | Value |
|---|---|
| Machine | Apple Silicon Mac (macOS), CPU execution (`DEVICE = cpu`; CUDA not available) |
| Python | 3.13.5 |
| Packages | torch 2.10.0, numpy 2.3.2, pandas 2.3.1, scikit-learn 1.7.1, scipy 1.16.1, matplotlib 3.10.5 |
| Runtime | 12.1 minutes inside the notebook (729 s including process start) |
| Seeds | split 42 · models 0, 1, 2, 3, 4 · validation noise 1234 · score estimators 2024 · bootstrap 7 · generation 0 |
| Dataset | `creditcard.csv`, SHA-256 `76274b69…98551a89` (matches the audited copy); all content checks passed |

The full record is in `colab/vae_outputs/environment.json` and `execution_report.json`.

## Phase-by-phase completion

| Notebook section | Result |
|---|---|
| 3–6 Setup, configuration, loading, verification | Dependencies present (none installed); dataset found and all 7 content checks passed |
| 7 EDA | Summary statistics and 4 figures |
| 8–9 Split and preprocessing | Train 199,366 (345 fraud) · validation 42,721 (74 fraud) · test 42,720 (73 fraud); 10 of 10 leakage checks passed |
| 10 Architecture | Forward-pass, shape and gradient checks passed (8,813 parameters) |
| 11–13 E1 training and threshold | 5 seeds trained and early-stopped (best epochs 103–246); thresholds from normal validation rows |
| 14–15 Generation and quality | 2 synthetic CSVs (42,647 rows each) and quality metrics |
| 16–17 Test detection and metrics | Predictions saved, then labels revealed; metrics, bootstrap confidence intervals, threshold sensitivity |
| 18–19 E2 and comparison | 5 E2 seeds trained, evaluated on the same test set; comparison table and figures |
| 20 Latent space and reconstruction | Latent and reconstruction figures; class-imbalance analysis |
| 21 Results | Markdown generated from this run's metrics; saved as `results_summary.md` |
| 23 Verification and export | 10 of 10 checks passed; `vae_outputs.zip` (74 files, 22.3 MB) passed the zip integrity test |

## Integrity verification (all passed)

| Check | Evidence |
|---|---|
| Final partition counts | 199,366 + 42,721 + 42,720 = 284,807; fraud 345 / 74 / 73 |
| `Class` never enters the model input | The model sees only the 29 features V1–V28 and log1p(Amount); `Class` and `Time` are absent from every input matrix |
| Primary model trains only on normal transactions | E1 training rows: 199,021, all normal |
| Threshold calibration uses no fraud labels | E1 τ comes from 42,647 normal validation rows; E2 τ from all validation rows, unlabelled |
| Test data used only for final evaluation | Label-access log: `split created; test labels locked` → `e1: test predictions saved` → `e1: test labels revealed` → `e2: test predictions saved` → `e2: test labels revealed` |
| Synthetic samples come from the decoder | Directly decoding z ~ N(0, I) with the same seed reproduces the exported generator output |
| Required plots exist and are valid | 20 PNG files, header and size checked |
| Export files are valid | Synthetic CSV: 42,647 rows, columns V1–V28 + Amount, no label, all finite. The checkpoint reloads into a fresh VAE and reproduces the test scores exactly. |
| Reported metrics match the saved files | `detection_metrics_per_seed.csv` re-read and compared with the in-memory values |

## Cross-check against the repository pipeline

The notebook re-implements the repository code inline. Run on CPU with the same seeds, it reproduces the repository results **exactly**:

- **Detection:** for all 10 models (E1 and E2 × 5 seeds), PR-AUC, ROC-AUC, precision, recall, F1, FPR, TP/FP/FN/TN, threshold, secondary-score APs and best epoch all match `outputs/metrics/detection_per_seed.csv` (maximum absolute difference 0.0). E1 mean PR-AUC is 0.43155445994067626 in both.
- **Synthetic data:** both CSVs are value-identical to `data/synthetic/`.
- **Synthetic-quality metrics differ by design.** The notebook uses normal *validation* rows as the real reference, so no test label is touched before predictions are saved, while the repo report uses normal *test* rows. Most metrics barely change (decoder mean: mean KS 0.221 in both; std ratio 0.50 vs 0.49; gradient-boosting two-sample AUC 1.00 in both). The **logistic two-sample AUC** is the exception: 0.78 vs 0.59. A follow-up over 3 seeds showed this is a stable reference-set effect (0.74–0.78 against validation rows, 0.58–0.59 against test rows, 0.50 for real vs real). The linear classifier is swayed by heavy-tailed outliers, so the reports present it as unstable and rely on the gradient-boosting test.

## Not verified here (needs a real Colab session)

- **Colab-specific code paths:** Google Drive search paths, `kagglehub` auto-download, the `files.upload()` prompt and the automatic `files.download()` of the zip. `kagglehub` is not installed locally, so that path printed a message and fell through to the local file as designed.
- **GPU execution:** the CUDA path selects the GPU automatically, but bitwise determinism isn't guaranteed there, so numbers can differ slightly in the last decimals.
- **Runtime on Colab:** will differ from the 12 minutes measured here. Expect it to be slower on a 2-vCPU runtime.

**Recommended:** run it once in Colab (Runtime → Run all) before submission, and confirm that section 23 reports all checks passed.
