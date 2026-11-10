# Dataset Comparison: Mendeley `ktbthg777x` v2 vs current `creditcard.csv`

Date: 2026-11-06
Status: evaluation only. No project data was replaced or modified.
Companion documents: [DATASET_AUDIT.md](DATASET_AUDIT.md) (current dataset), [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Recommendation: keep `creditcard.csv` as the project dataset. Do not switch to the Mendeley dataset.** The evidence is in §9.

All candidate statistics below were computed during this evaluation from the downloaded file. The download and the analysis scripts are in the session scratchpad (`/private/tmp/claude-501/-Users-vansh-Downloads-genAI/8da0c31f-4039-411f-820b-20aaaf9a5ab7/scratchpad/`), outside the project folder.

---

## 1. Source, publication, licence, provenance

| Item | Finding | How verified |
|---|---|---|
| Title | *Synthetic Banking Transaction Dataset with Multi-Pattern Fraud Labels for Machine Learning Research* | Mendeley public API |
| DOI | 10.17632/ktbthg777x.2 | API |
| Authors | Md Mehedi Hasan (ORCID 0009-0001-2150-0288), Maria Rashid (ORCID 0009-0002-5165-797X), Daffodil International University, Bangladesh | API |
| Versions | v1 published 2025-10-31; **v2 published 2025-11-03** | API |
| Licence | **CC BY 4.0**. Academic use is allowed with attribution. | API |
| Nature of data | **Fully synthetic.** Generated with NumPy, Pandas, and the Faker library, according to the authors' description. | API description, confirmed by the data (fake city names, random coordinates) |
| Associated paper | **None linked.** The record lists no articles or related links. | API (`articles: []`) |
| Generator code | **Not published.** The description calls the dataset "configurable" and "reproducible", but no script, seed, or parameters are provided. | API, file listing |
| Files | One zip, `Fraud Detection Dataset-bmJaU4.zip` (62,515,239 bytes), containing `Fraud Detection Dataset.csv` (175,509,007 bytes) | API listing, `unzip -l` |
| Integrity | Zip SHA-256 `8ab858b4…9958d8` **matches** the published hash. CSV SHA-256: `e3895c98…31caf6`. | `shasum` |
| v1 vs v2 | v1 and v2 contain the **same file** (identical SHA-256). v2 changed only the metadata. | API |
| Community use | 0 recorded views and 0 downloads at the time of checking | API metrics |

### Description vs actual file

| The description says | The file contains |
|---|---|
| "Fraud rate: ~0.5%" | **0.149%** (1,490 / 1,000,000) |
| "CSV and Parquet" | CSV only |
| "100,000 unique customer profiles" | 99,996 distinct `customer_id` values |
| "20,000 merchant entities" | 20,080 distinct `merchant_id` values: 20,000 real merchants plus 80 pseudo-merchants named `TRANSFER_TO_CUST…` |
| "Temporal range: 1 year" | 2024-10-30 20:56 to 2025-10-30 20:54 UTC (364 days). Matches. |

**Provenance summary:** the source is identifiable, the licence is clear, and the file integrity checks out. But the dataset was published 2025-11, has no peer-reviewed paper and no generator code, has no recorded community use, and its description does not match its contents on fraud rate and file formats. Academically, its provenance is weak.

---

## 2. Schema and data dictionary

The authors provide **no formal data dictionary**. The table below is inferred from the data itself.

| # | Column | Type | Unique values | Notes |
|---|---|---|---|---|
| 1 | `transaction_id` | ID string `TXN` + 12 digits | 1,000,000 | Row identifier. Order is uncorrelated with time (Spearman ρ = 0.0007), so rows are shuffled. |
| 2 | `timestamp` | ISO 8601 UTC, 1-second resolution | 984,478 | 15,522 repeated timestamp values |
| 3 | `customer_id` | ID `CUST` + 7 digits | 99,996 | About 10 rows per customer (range 1–27) |
| 4 | `card_id` | ID `CARD<customer#>-<n>` | 238,624 | The card number embeds its owner's customer number (see §4) |
| 5 | `device_id` | 12-character hex | 196,942 | About 2 devices per customer; 77 devices are shared across customers |
| 6 | `ip_address` | IPv4 string | 980,372 | Almost unique per row, so it carries no information |
| 7 | `merchant_id` | ID `MID` + 6 digits | 20,080 | Includes 80 `TRANSFER_TO_CUST…` pseudo-IDs |
| 8 | `merchant_category` | categorical | 11 | 9 categories at about 10.6–10.9% each, `online_marketplace` 3.0%, `transfer` 80 rows |
| 9 | `merchant_country` | categorical (ISO-2) | 10 | AU, BR, IN, CA, ES, US, IT, GB, DE, FR, each about 9.6–10.5% |
| 10 | `merchant_city` | free text (Faker) | 13,333 | Fake names such as "East John" and "Port Jennifer", not tied to country |
| 11 | `merchant_latitude` | float | 20,000 | Uniform on [−60, 70], **unrelated to country** |
| 12 | `merchant_longitude` | float | 20,000 | Uniform on [−180, 180], **unrelated to country** |
| 13 | `transaction_type` | categorical | 2 | `purchase` 999,920; `transfer` 80 |
| 14 | `amount` | float, 2 dp | 34,573 | 1.00–5,000.00 USD |
| 15 | `currency` | categorical | 1 | Always `USD`, so a constant column |
| 16 | `is_fraud` | int 0/1 | 2 | **Target** |
| 17 | `fraud_type` | categorical / empty | 5 | Empty for legitimate rows; one of 4 fraud types otherwise |

**There is no `fraud_probability` column** in this dataset, nor any risk score or other "risk" field, in either version. If such a column appears in some other copy of this data (a Kaggle re-upload, a derivative), it would almost certainly be an output of the generator's labelling process, and would have to be dropped as direct target leakage.

---

## 3. Data-quality audit

| Check | Result |
|---|---|
| Missing values | 0 in all columns except `fraud_type`, which is empty for all 998,510 legitimate rows (by design) |
| Duplicate rows | 0 (also 0 when ignoring `transaction_id`, and 0 when ignoring `transaction_id` and `timestamp`) |
| Duplicate `transaction_id` | 0 |
| Infinite / unparsable numbers | 0 |
| Invalid coordinates | 0 outside the valid range. However, coordinates are **semantically invalid**: every country spans latitudes −60 to 70 and the full longitude range. |
| Invalid IPs | 0 (all octets ≤ 255) |
| `amount` | No zero or negative values. Minimum exactly 1.00 (5,534 rows); maximum exactly 5,000.00 (23 rows). Both bounds look like generator clipping. |
| Constant columns | `currency` (always USD) |
| Near-constant columns | `transaction_type` (99.992% `purchase`) |
| Entity consistency | A merchant always keeps the same category and country. 407 merchants have more than one location/city, which comes from fraud injection (§4). |

**Class distribution:** 998,510 legitimate and 1,490 fraud (**0.149%**). Fraud types:

| `fraud_type` | Rows |
|---|---|
| `card_testing` | 800 |
| `geo_anomaly` | 400 |
| `account_takeover` | 210 |
| `money_laundering_ring` | 80 |

The round counts (800/400/210/80) show that fraud was injected in fixed quantities, not produced by a behavioural process.

### Distributions and dependence (legitimate rows)

- **`amount`:** log(amount) has mean 3.00, std 1.10, skew 0.06, excess kurtosis 0.27. That is effectively a **log-normal distribution clipped to [1, 5000]**. Raw skew is 18.3; after `log1p` it is 0.28.
- **Amount does not depend on category, country, or hour.** Median amount is 19.97–20.27 in every category and 19.89–20.24 in every country. Kruskal–Wallis p-values: category 0.28, country 0.42, hour 0.92.
- **Category is independent of country:** Cramér's V = 0.022.
- **Coordinates are independent of everything:** Spearman ρ(amount, lat) = 0.0007, ρ(amount, lon) ≈ 0.0000, ρ(lat, lon) = −0.005.
- **Time:** about 41,000–42,000 transactions in **every** hour of the day, about 142,000–144,000 on every weekday, and about 82,000–85,000 per full month. There is no daily, weekly, or seasonal pattern. The fraud rate is flat across months (0.13–0.16%) and hours.
- **The one real structure:** customers differ in typical spend. The standard deviation of per-customer mean log-amount is 0.75, against 0.35 expected if amounts were independent across customers. Each customer buys from about 6.3 countries and 4.0 categories.

**Conclusion:** a legitimate row is essentially a draw from a **product of independent marginal distributions** (log-normal amount × uniform category × uniform country × uniform coordinates × uniform time), plus a per-customer spending level.

---

## 4. Target leakage and shortcut features

| Field / pattern | Relationship to the label | Severity |
|---|---|---|
| `fraud_type` | Non-empty **if and only if** `is_fraud = 1` | **Critical, direct leakage. Must be dropped.** |
| `transaction_type = transfer` | 80/80 are fraud (precision 1.0, recall 5.4%) | **Critical generator artefact** |
| `merchant_category = transfer` | Same 80 rows | Critical generator artefact |
| `merchant_id` starting `TRANSFER_TO_` | Same 80 rows; it also encodes the receiving customer's ID | Critical generator artefact |
| **`card_id` customer number ≠ `customer_id`** | 880 rows, **all fraud**: every `card_testing` row and every `money_laundering_ring` row. A single deterministic rule achieves **precision 1.0 at recall 0.59**. | **Critical generator artefact.** 59% of all fraud is identifiable from one string comparison. |
| Device first seen for the customer | 100% of `account_takeover`, `card_testing`, and `money_laundering_ring` rows, against 19.6% of legitimate rows | Weak to moderate signal. It is computable without labels but needs history features. |
| Shared device across customers | 77 of 80 `money_laundering_ring` rows | Artefact |
| `fraud_probability` / risk fields | **Not present** in this dataset | N/A |

The four fraud types are written by injection rules. Two of them (card testing and money laundering) are marked by **identifier artefacts** rather than by behaviour. A detector that "finds" them is detecting the generator's bookkeeping, not fraud. A defensible study would have to drop or neutralise `card_id`, `merchant_id`, `transaction_type`, and `merchant_category = transfer`. Doing that removes most of the detectable fraud.

---

## 5. Suitability for normal-only VAE training and synthetic generation

### 5.1 What a row-level VAE could learn

High-cardinality identifiers (customer, card, device, IP, merchant, city) cannot be VAE inputs. The usable row-level features are `amount`, `merchant_category`, `merchant_country`, latitude, longitude, hour, and day of week. In legitimate data these are **mutually independent** (§3). The VAE's latent space would therefore have no joint structure to capture. The best it could do is reproduce each marginal distribution on its own, which a histogram does equally well.

### 5.2 Empirical ceiling for row-level detection

As a proxy for what any row-level density model, a VAE included, could achieve, we fitted a **label-free independent-marginals model** on legitimate rows of a random 70% training split: a log-normal for amount plus empirical frequencies for category, country, transaction type, and hour. Its negative log-likelihood was used as the anomaly score on the held-out 30%.

| Score / subset | Average Precision | Base rate (random scorer) |
|---|---|---|
| All test rows | 0.071 | 0.0015 |
| Excluding the 80 `transfer` rows | **0.0085** | 0.0015 |
| `card_testing` vs legitimate | 0.011 | 0.0008 |
| `geo_anomaly` vs legitimate | **0.0005** | 0.0004 (no better than chance) |
| `account_takeover` vs legitimate | **0.0004** | 0.0002 (no better than chance) |
| `money_laundering_ring` vs legitimate | 1.0 | 0.0001 (only because "transfer" is an unseen category) |

Almost all of the row-level signal comes from the 80 `transfer` artefact rows. **Geographic anomalies and account takeovers are invisible at row level.** Coordinates are random everywhere, so a "geo anomaly" is just another random coordinate. Account takeover differs only in entity history. Card testing uses small amounts, but about 10% of legitimate amounts are equally small.

Detecting these types would need per-customer or per-merchant **sequence and history features** such as device novelty, merchant location drift, or velocity. That turns the project into a feature-engineering and graph or sequence problem, away from the VAE-on-tabular-data design in the proposal. Even then, card testing is not temporally bursty here: the median gap from the previous transaction on the same card is about 19 days. So the injected patterns do not behave like the real-world patterns they are named after.

### 5.3 Synthetic generation

- Training a generative model on data that is **already synthetic** produces "synthetic data from synthetic data". It approximates a simple NumPy/Faker script, which is hard to defend academically as learning the distribution of financial transactions.
- Fidelity checks would be close to trivial. Every correlation is about 0, so a correlation-matrix comparison has nothing to measure, and an independent-marginals sampler would match a VAE.
- Mixed types (two continuous features plus categorical features of 10–11 levels) would need a mixed-likelihood decoder (Gaussian + categorical). That adds complexity without adding structure to learn.

**Verdict:** technically usable, but **unsuitable for the project's scientific goals** in both normal-only anomaly detection and generation.

---

## 6. Can a reproducible 200,000-row subset preserve class and temporal characteristics?

**Yes, mechanically.** Two options were tested:

| Subset method | Rows | Frauds | Fraud % | Fraud types (card testing / geo / ATO / ML) | Temporal check | Problem |
|---|---|---|---|---|---|---|
| **A. Row-stratified** by (`fraud_type` × month), seed 42 | 200,000 | 298 | 0.149% (exact) | 162 / 79 / 42 / 15 | KS(timestamp) p = 0.99996; KS(hour) = 0.002; KS(amount) = 0.001 | Breaks entity histories: about 2.3 rows per customer instead of 10, so device-novelty and other history features become meaningless |
| **B. Customer-grouped** (whole customers until about 200k rows) | 199,981–200,000 over 10 seeds | 268–330 | 0.134–0.165% (mean 0.150%) | seed 42: 165 / 82 / 50 / 17 | KS(timestamp) = 0.002 | Keeps customer histories, but breaks money-laundering links (only 0–27% of receivers are in the subset). The class rate varies by ±0.011 percentage points across seeds. |

Preserving the "temporal characteristics" is easy only because the dataset **has almost none**: timestamps are uniform across hours, weekdays, and months. A subset that matches the full data on these tests is not evidence of realistic temporal behaviour.

If the candidate were ever used: Option A (fixed seed, stratified by type × month, with history features computed on the full 1M rows *before* subsetting) preserves the label distribution exactly. Option B is preferable if history features have to be computed inside the subset.

---

## 7. Preprocessing the candidate would need (if adopted)

| Feature group | Recommended handling |
|---|---|
| **Drop: leakage and artefacts** | `fraud_type`; `transaction_type`, `merchant_category = transfer`, and the `TRANSFER_TO_` merchant IDs, either removed or explicitly reported as an artefact subset; any feature derived from the card number's customer prefix |
| **Drop: no information** | `currency` (constant), `transaction_id`, `ip_address` (about 98% unique), `merchant_city` (Faker text), raw `customer_id`, `card_id`, `device_id`, `merchant_id` |
| Numerical | `amount` → `log1p` → standardise using legitimate training rows only. Latitude and longitude → scale to [−1, 1] or project to 3-D unit-sphere coordinates. (They are random, so including them only adds noise.) |
| Categorical | `merchant_category` (10 after removing `transfer`) and `merchant_country` (10) → one-hot. The decoder needs softmax/categorical cross-entropy heads plus a Gaussian head for continuous features. Unseen categories at test time map to an "other" bucket. |
| Temporal | Hour and day of week → sin/cos cyclical encoding. Never feed raw timestamps. Use a **time-ordered split** (train on earlier months, test on later ones), because a row-shuffled split leaks future entity history. |
| History features (optional) | Per customer: is this a new device, time since previous transaction, amount z-score against the customer's past amounts. Compute these **causally**, using only past rows, so no future information enters a row's features. |
| Split | Group by customer, or split by time, to avoid entity leakage across train and test |

---

## 8. Side-by-side comparison

| Criterion | `creditcard.csv` (current) | Mendeley `ktbthg777x` v2 |
|---|---|---|
| Origin | Real European card transactions (Worldline / ULB, Sept 2013), anonymised with PCA | **Fully synthetic** (NumPy + Faker) |
| Provenance | Long-established public benchmark, cited in thousands of papers (Dal Pozzolo et al., 2015) | Published 2025-11; no paper, no generator code, no recorded downloads; description does not match the data |
| Licence | ODbL 1.0 (database) / DbCL 1.0 (contents), verified via the Kaggle API; see DATASET_AUDIT.md §1 for open ambiguities | CC BY 4.0 (verified via API) |
| Size | 284,807 × 31 (144 MB) | 1,000,000 × 17 (176 MB CSV, 63 MB zipped) |
| Fraud rate | 0.173% (492) | 0.149% (1,490) |
| Fraud mechanism | Real fraud, labelled after the fact | 4 injected rule-based patterns in fixed counts |
| Direct leakage fields | None among inputs (only `Class`) | `fraud_type`; transfer fields; card/customer ID mismatch (59% of fraud) |
| Structure in legitimate data | Real dependencies, e.g. Amount–V2 \|r\| = 0.53 and Amount–V7 0.40; heavy tails | Features **mutually independent** (Cramér's V 0.02; all ρ ≈ 0) |
| Row-level fraud signal | Strong: fraud means shifted 5–9σ on V17/V14/V12/V10; 83% of frauds have a \|z\| > 5 feature | Weak apart from artefacts: AP 0.0085 excluding transfer rows; geo/ATO at chance |
| Feature interpretability | Low (anonymised V1–V28) | High (amount, category, country, time). The main advantage. |
| Temporal realism | 48 h with a visible day/night cycle and night-time fraud peaks | 1 year with no daily, weekly, or seasonal pattern |
| Fit to the VAE design | All-continuous, so a single Gaussian decoder works | Mixed types need a mixed decoder; real signal needs sequence/history engineering |
| Generative-modelling value | Learning an unknown real distribution: meaningful | Re-learning a simple random generator: weak |
| Comparability with literature | High | None |
| Preprocessing effort | Low–moderate (dedupe, log Amount, drop Time, scale) | High (leakage removal, ID handling, encoding, causal history features) |
| Privacy | Already anonymised | No privacy concerns |
| Fits the 16 GB Mac | Yes (~71 MB in memory) | Yes, but about 1 GB as strings in pandas; needs typed loading or a subset |

### Advantages of the candidate
1. Interpretable, human-readable features. These would make a nicer Streamlit demo and could address the proposal's mention of "location, frequency".
2. Larger, with about 3× more fraud examples and fraud-type sub-labels for per-type error analysis.
3. A permissive CC BY 4.0 licence and no privacy concerns.
4. A one-year time span.

### Limitations of the candidate
1. Fully synthetic with independent features. There is nothing real for a generative model to learn, and synthetic-from-synthetic is hard to justify academically.
2. Shortcut and leakage artefacts mark most of the fraud (59% by one rule, 5% by a "transfer" category).
3. The remaining fraud types are undetectable from row-level features.
4. Weak provenance, and the description does not match the contents (stated 0.5% fraud vs actual 0.149%; Parquet missing; no data dictionary).
5. No temporal realism despite the one-year span.
6. Results would not be comparable with any published work.

---

## 9. Recommendation

**Keep `creditcard.csv` as the sole primary dataset.** Evidence:

1. **Scientific validity of the generative task.** The current data holds real, unknown dependencies, such as Amount–V2 |r| = 0.53 and heavy-tailed, correlated structure within the legitimate class, that a VAE must learn. The candidate's legitimate rows are a product of independent marginals (§3, §5.1), so a VAE cannot outperform a histogram sampler on them.
2. **Validity of the anomaly-detection task.** In the current data, fraud is separable in feature space: mean shifts of 5–9σ, and 83% of frauds have an extreme feature value. In the candidate, once the artefacts are removed, label-free row-level detection is at chance for 2 of the 4 fraud types (§5.2), and most of the "detectable" fraud is identifier bookkeeping (§4).
3. **Integrity.** The candidate has a perfect leakage field (`fraud_type`) and deterministic shortcut rules covering 59% of all frauds (880 card-mismatch rows; the 80 transfer rows are a subset of these), and its documentation contradicts its data.
4. **Academic defensibility.** The ULB dataset is a recognised benchmark with clear provenance. The candidate has no paper, no code, and no usage history.
5. **Fit to the approved plan.** No changes to the methodology, architecture, or phases in PROJECT_PLAN.md are needed.

**Optional, limited use (only if you want it, and not as a replacement):** the candidate could appear in the report's limitations or discussion section as a negative example. It shows why synthetic benchmarks with independent features and injected, ID-marked fraud are poor tests for density-based detectors. This would cost about one notebook and is not needed for the core deliverables.

No files were switched, replaced, or added to the project data folders. Waiting for your decision.
