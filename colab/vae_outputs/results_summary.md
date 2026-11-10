
### Anomaly detection (test set: 42,647 normal, 73 fraud)

| | E1 normal-only (primary) | E2 unsupervised |
|---|---|---|
| PR-AUC (average precision) | **0.432 ± 0.047** | 0.038 ± 0.016 |
| ROC-AUC | 0.947 ± 0.016 | 0.863 ± 0.016 |
| Precision at τ | 0.104 ± 0.013 | 0.042 ± 0.003 |
| Recall at τ | 0.652 ± 0.072 | 0.263 ± 0.025 |
| F1 at τ | 0.179 ± 0.021 | 0.073 ± 0.006 |
| False-positive rate | 0.010 ± 0.001 | 0.010 ± 0.001 |

**E1 performance.**
- **Overall:** the one-class VAE detects fraud without ever seeing a fraud label. Its PR-AUC of 0.432 is about 253× the random baseline (0.0017).
- **Threshold:** the validation-calibrated τ gives a test false-positive rate of 0.0097, close to the 1% target, so the label-free calibration transfers to unseen data.
- **Headline seed 0:** 55 of 73 frauds caught and 413 false alarms (precision 0.118, 95% CI 0.100–0.134; recall 0.753, CI 0.644–0.849). The best seed by AP is seed 0, and the 5-seed mean is the representative figure.
- **Precision is low** because of class imbalance. Even at a 0.97% false-positive rate there are about 7.5 false alarms per fraud caught. At a 1% or 10% fraud rate, the same detector would reach precision of about 0.44 and 0.90 (see the class-imbalance figure).

**Contamination (E2).**
- **The gap:** training on all rows lowers PR-AUC from 0.432 to 0.038; E1 is better in 5 of 5 seeds (paired bootstrap ΔAP 0.462, 95% CI 0.351–0.575).
- **The cause:** the training frauds are 0.17% of rows but 4.1% of the squared feature magnitude, so squared-error training spends capacity on reconstructing them. Frauds also resemble each other, so this carries over to unseen frauds: the median test-fraud error falls from 2.08 (E1) to 0.86 (E2), and the fraud/normal error ratio falls from 5.7× to 2.5×.
- **The latent space still separates frauds:** centroid distance 4.69 for E1 and 3.86 for E2. Contamination mainly damages the *reconstruction* score, which is why E2's ELBO-based score (AP 0.117) beats its reconstruction score.

**Reconstruction and anomaly scores.**
- **What the score measures:** reconstruction error shows how far a transaction is from what the model learned as normal. Frauds that look structurally normal are missed, and rare legitimate transactions raise false alarms. A high score is **not** a fraud probability.
- **Secondary scores:** the negative ELBO (AP 0.450) and the importance-weighted −log p(x) (AP 0.420) rank transactions similarly.

### Synthetic data (E1 seed 0, 42,647 rows per sampler)

| | Decoder mean (primary) | + fitted noise (variant) | Real vs real |
|---|---|---|---|
| Mean KS | 0.221 | 0.053 | 0.006 |
| Std ratio (V features) | 0.50 | 0.85 | 1.00 |
| Correlation difference (Frobenius) | 3.73 | 1.69 | 0.66 |
| Amount–V2 correlation (real -0.55) | -0.56 | -0.17 | -0.55 |
| Classifier two-sample AUC, logistic / boosting | 0.77 / 1.00 | 0.55 / 0.99 | 0.50 / 0.50 |
| Exact copies of training rows | 0 | 0 | — |

**Assessment:**
- **Validity:** all samples are valid (100%) and none copies a training row.
- **Primary sampler:** decoder-mean samples match the centre and the main Amount relationships, but they are too smooth (about 50% of the real spread).
- **Noise variant:** adding fitted noise brings per-feature distributions much closer, at the cost of diluting dependencies.
- **Detectability:** a non-linear classifier still separates both samplers from real data, so the VAE captures first-order structure but not the full joint distribution.
- **Interpretability:** samples live in the anonymised PCA feature space and are not interpretable transactions.
