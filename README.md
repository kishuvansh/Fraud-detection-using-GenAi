# VAE-Based Financial Transaction Modelling

IT429 Generative AI project. A Variational Autoencoder built from scratch in PyTorch for:
- **synthetic transaction feature-vector generation**, and
- **one-class anomaly detection** on credit-card transactions.

**Start here:** [Final report](reports/final_report.md) · [Colab notebook](colab/VAE_Financial_Transaction_Modeling_Colab.ipynb) · [Notebook execution report](reports/notebook_execution_report.md)

**More docs:** [PROJECT_PLAN.md](PROJECT_PLAN.md) · [DATASET_AUDIT.md](DATASET_AUDIT.md) · [Detailed results](reports/vae_results.md) · [EDA and data preparation](reports/eda_and_data_preparation.md)

## Headline results (test set: 42,647 normal + 73 fraud transactions, never used for any decision)

| | Normal-only VAE (E1, primary) | Unsupervised VAE (E2) |
|---|---|---|
| Average precision (PR-AUC), 5 seeds | **0.432 ± 0.047** (random = 0.0017) | 0.038 ± 0.016 |
| ROC-AUC | 0.947 ± 0.016 | 0.863 ± 0.016 |
| Recall / precision at the validation-calibrated threshold (~1% FPR) | 0.65 / 0.10 | 0.26 / 0.04 |

**Synthetic data:** 42,647 synthetic normal-transaction feature vectors per sampler, in `data/synthetic/`. The pre-registered decoder-mean sampler keeps the main Amount–V correlations but has about half the real spread. The fitted-noise variant matches per-feature distributions much better (mean KS 0.05 vs 0.22) but weakens the Amount–V correlations. Neither is a memorised copy of training data. A non-linear classifier can still tell both apart from real data. Full assessment: [reports/vae_results.md](reports/vae_results.md).

## Status

| Work | State |
|---|---|
| EDA, leakage-safe split and preprocessing (notebooks 01–02) | Done |
| VAE implementation, training of E1 and E2 (5 seeds each), threshold calibration (notebook 03) | Done |
| Synthetic generation and quality analysis (notebook 04) | Done |
| Test-set anomaly detection, threshold sensitivity (notebook 05) | Done |
| Secondary experiment comparison, latent space, reconstruction, class imbalance, limitations (notebook 06) | Done |
| Self-contained Colab notebook, verified top to bottom in a fresh process (`colab/`) | Done |
| Architecture sensitivity analyses, Streamlit app, slides | Not started |

## Google Colab (one notebook, everything)

1. Open `colab/VAE_Financial_Transaction_Modeling_Colab.ipynb` in Colab. Optionally select a GPU runtime.
2. Provide `creditcard.csv` in one of three ways: upload it with the file browser, put it in `MyDrive/` with Drive mounted, or let the notebook try `kagglehub` or prompt for an upload.
3. Choose **Runtime → Run all**. No cell needs editing. Outputs go to `vae_outputs/` and are downloaded as `vae_outputs.zip`: checkpoints, synthetic CSVs, metrics, plots, an execution report and the environment record.

## Setup

- **Data:** place `creditcard.csv` in the project root. Its SHA-256 is checked on every load.
- **Locally:** `pip install -r requirements.txt` (Python 3.11+).
- **On Google Colab:**
  1. Upload the project folder to Drive (default path: `MyDrive/genAI`).
  2. Run the notebooks in order. Notebook 03 trains any missing checkpoints, and notebooks 04 and 05 run their pipeline scripts if their outputs are missing.

## Reproduce

Run these from the project root. Training is about 70 s per seed on a CPU.

```bash
python scripts/run_eda.py                                   # EDA statistics and figures
python scripts/prepare_data.py                              # split, scalers, leakage checks
python scripts/train_vae.py --experiment e1_primary         # 5 seeds; add --device auto for CUDA/MPS
python scripts/train_vae.py --experiment e2_unsupervised
python scripts/generate_synthetic.py                        # synthetic CSVs + quality metrics/figures
python scripts/evaluate_detection.py                        # test evaluation (predictions saved before labels are read)
python scripts/analyse_models.py                            # E1 vs E2 comparison, latent space, reconstruction, imbalance
python tests/test_vae.py && python tests/test_training.py && python tests/test_data_pipeline.py
python scripts/run_notebook.py notebooks/0*.ipynb           # re-execute notebooks without Jupyter
```

## Layout

```
configs/                 data.yaml, e1_primary.yaml, e2_unsupervised.yaml
src/data/                loading, EDA statistics, split, preprocessing, leakage checks
src/models/vae.py        the VAE (encoder, reparameterization, decoder, prior sampling)
src/training/            loss (ELBO), training loop, run orchestration
src/generation/          synthetic sampling and CSV export
src/detection/           anomaly scores, threshold calibration, ordered test evaluation
src/evaluation/          metrics and figures
scripts/                 command-line entry points + notebook runner
tests/                   26 tests (model, training, data pipeline)
notebooks/               01 EDA · 02 preparation · 03 model and training · 04 generation · 05 detection · 06 comparison and analysis
colab/                   self-contained end-to-end Colab notebook (+ its verified run outputs)
data/processed/          splits and fitted transformers
data/synthetic/          generated CSVs (+ metadata and README)
outputs/runs/            per-run checkpoints, histories, thresholds, scores, predictions, metrics
outputs/figures/         eda / training / generation / detection
outputs/metrics/         aggregated results
reports/                 written summaries
```

## Dataset and attribution

*Credit Card Fraud Detection* dataset, Machine Learning Group – ULB / Worldline, Kaggle (`mlg-ulb/creditcardfraud`), ODbL 1.0 / DbCL 1.0. Please cite: Dal Pozzolo, Caelen, Johnson & Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015. The raw CSV is not included in this repository.

Synthetic samples are vectors in the dataset's anonymised PCA feature space (`V1`–`V28` plus `Amount`). They are not interpretable real-world transaction records, and they carry no fraud label. Anomaly scores measure how unusual a transaction is relative to learned normal behaviour; they are not fraud probabilities.
