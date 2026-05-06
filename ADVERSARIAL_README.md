# Adversarial DoH C2 Evasion Framework

**Branch:** `adversarial` | **Role:** Attacker  
**Project:** DoH Tunnel / C2 Detection | Network Security Final Project  
**Dataset:** Hokkaido University Combined Dataset (CIRA-CIC-DoHBrw-2020 + DoH-Tunnel-Traffic-HKD)

---

## Overview

This branch implements the **adversarial side** of the project. The goal is to generate DoH C2 traffic that evades the flow-based ML classifiers built in the `detectors` branch (RF, GB, XGB, MLP).

We built three pipelines, each representing a different attacker capability level:

| Pipeline | Attacker Knowledge | Purpose |
|---|---|---|
| `integrate_detectors.py` | Synthetic features, no real packets | Fast iteration baseline |
| `real_adversarial_pipeline.py` | Full model access + scaler | White-box upper bound |
| `blackbox_adversarial_pipeline.py` | Only observed benign traffic | Realistic attacker |

---

## Key Findings

### White-box Attack: Upper Bound
*Attacker has: scaler, feature names, predict_proba, SHAP analysis*

| Strategy | RF | GB | XGB |
|---|---|---|---|
| naive | 100% | 100% | 100% |
| timing_only | 100% | 100% | 100% |
| size_mimicry | 100% | 100% | 100% |
| cover_traffic | 100% | 100% | 100% |
| full_mimicry | 100% | 100% | 100% |
| adaptive | 100% | 100% | 100% |

### Black-box Attack: Realistic Attacker
*Attacker has: observed benign traffic, CICFlowMeter, Scapy. No scaler, no model internals.*

| Strategy | RF | GB | XGB |
|---|---|---|---|
| naive | 0% | 0% | 0% |
| timing_only | 0% | 40% | 0% |
| size_mimicry | 0% | 40% | 0% |
| cover_traffic | 0% | 25% | 0% |
| full_mimicry | 0% | 50% | 0% |
| adaptive | 0% | 50% | 0% |

### The Critical Finding

The gap between white-box (100%) and black-box (0-50%) reveals that **the scaler (`scaler.joblib`) is the key defense artifact**. Without it, RF and XGB are completely robust. GB is partially vulnerable.

This means:
- If the defender keeps the scaler private, RF and XGB are strong defenses
- GB has inherent weaknesses even without model access
- An attacker who steals the scaler gains a massive advantage

---

## Threat Model: Three Levels

### Level 1: Black-box (Realistic)
- Attacker observes benign DoH traffic on their network
- Uses public tools (CICFlowMeter, Scapy)
- Gets binary feedback: blocked or not blocked
- **Result: 0-50% evasion (GB only)**

### Level 2: Gray-box (Partial knowledge)
- Attacker has CIRA-CIC benign distributions from published literature
- Knows general feature structure from papers
- Does not have the scaler
- **Result: Similar to black-box without scaler**

### Level 3: White-box (Upper bound)
- Attacker has full model access, scaler, feature names, SHAP
- Represents a worst-case scenario (model theft)
- **Result: 100% evasion across all strategies and models**

---

## Why It Works (White-box): SHAP Analysis

We ran SHAP on all three models to identify which features drive detection:

| Feature | RF Importance | XGB Importance |
|---|---|---|
| PacketLengthMode | 0.124 | 5.624 |
| PacketLengthMedian | 0.035 | 1.344 |
| FlowBytesReceived | 0.037 | 1.518 |
| PacketLengthMean | 0.037 | 1.012 |
| ResponseTimeTimeMean | 0.008 | 0.497 |

**Key insight:** Real benign DoH has tiny packets (mode ~74B). C2 tools send large uniform packets. Match the size distribution and fool the classifier.

**Two critical bugs we fixed:**
1. CICFlowMeter outputs time in **seconds**, but the training data used **milliseconds**, so multiply x 1000
2. Benign flows have 400-600 packets (~62KB), but we initially generated only ~50

---

## Architecture

```
traffic_shaper.py
      | (6 evasion strategies)
[optional] benign_target_sampling.py
      | interpolated (kNN blend) or correlated (Ledoit–Wolf MVN) or legacy wire stats
      | -> traffic_params_from_cira_features() drives IAT / sizes in schedule_to_session()
[Scapy] - Full TCP sessions with handshake + data + teardown
      |
PCAP file (temp dir; CICFlowMeter in-process on Windows, CLI fallback)
      |
[CICFlowMeter] - flow features (mapped to detector columns via map_features)
      |
[optional] realism filter: mapped features vs CIRA benign marginals (not identical feature space)
      |
[White-box] scaler.transform() + predict_proba()
[Black-box] raw unscaled features + model.predict() -> 0-50% typical evasion
```

### Realistic benign targeting and validation (pipeline v5)

The white-box pipeline can draw **CIRA L2–style benign targets** (same statistical columns as the reference dataset), map them to **Scapy session parameters**, then run **CICFlowMeter** on synthetic PCAPs. That chain does **not** reproduce raw CIRA rows exactly (CIC ≠ CIRA; `map_features` is approximate), so validation is split:

| Layer | What it checks |
|---|---|
| **Tabular correlation validation** (`--correlation-validation`) | Pearson **Corr(real benign)** vs **Corr(N synthetic CIRA rows)** from the active sampler. Writes `correlation_validation_report.json` (Frobenius norm of matrix difference, max entry error, mean off-diagonal error). **CIRA space only.** |
| **Marginal realism filter** (`--realism-filter`) | After CIC + `map_features`, compares detector features to **benign quantile bands + z-scores** on the reference table. Use `--realism-pass any` for cross-domain tolerance (default). |
| **`--validate`** | Prints a short **CIC column vs benign** comparison for the **first successful CIC extraction** on **flow index 0** within each strategy (does **not** wait for `--realism-filter` acceptance). Uses **`data/l2-total-add.csv`** only (hardcoded), not `--reference-data`; if that CSV is missing, validation prints nothing. |

**Target sampling modes** (`--target-sampling`):

| Value | Behavior |
|---|---|
| `interpolated` (default) | kNN-style blend of two nearby benign rows + noise + quantile clip (same spirit as `synthetic_realistic_doh_generator.ipynb`). |
| `correlated` | Multivariate Gaussian in standardized space with **Ledoit–Wolf** shrunk covariance; inverse-transform and clip. Explicit linear correlations among CIRA columns. |
| `legacy` | Original v4-style random wire stats (no CIRA row sampling). |

Reference data auto-resolves to `L2-BenignDoH-MaliciousDoH.parquet` or `data/l2-total-add.csv` when present; override with `--reference-data`.

**Environment:** `DOH_CFM_FORCE_CLI=1` forces CICFlowMeter CLI instead of in-process extraction. `DOH_CFM_DEBUG=1` prints failures when falling back.

---

## Evasion Strategies

| Strategy | Description |
|---|---|
| `naive` | Fixed 5s beacon, no shaping (baseline, trivially caught) |
| `timing_only` | Randomized IAT matching browser Poisson distribution |
| `size_mimicry` | Timing + packet sizes matching benign distribution |
| `cover_traffic` | Decoy queries to real resolvers (Google, Cloudflare) |
| `full_mimicry` | Timing + size + cover + burst grouping to simulate page loads |
| `adaptive` | Full mimicry + classifier feedback loop |

---

## Setup

### Dependencies

**White-box / black-box pipelines** (`real_adversarial_pipeline.py`, `blackbox_adversarial_pipeline.py`), including benign sampling and CIC extraction:

```bash
pip install numpy pandas scipy scikit-learn joblib xgboost scapy cicflowmeter
```

- **`xgboost`** — Required so `joblib` can load **`xgb.joblib`** (same as training). RF and GB use **scikit-learn** only.
- **`pyarrow`** (or **`fastparquet`**) — Optional; needed to read **`*.parquet`** reference files (`--reference-data`, default `L2-*.parquet`). CSV-only workflows do not need it.

**Train detectors** (`detector.py`, Step 1) additionally need what's imported there:

```bash
pip install matplotlib seaborn shap torch
```

(Often installed alongside the same base stack; neural-net training uses **`torch`** only when you omit `--no_nn`.)

### Fix CICFlowMeter v0.5.0 Bugs

```bash
python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/sniffer.py'
txt = open(path).read()
fixed = txt.replace('if fields is not None:', 'if fields is not None and fields is not False and fields != True:')
open(path, 'w').write(fixed)
print('Fixed sniffer.py')
"

python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/flow.py'
txt = open(path).read()
fixed = txt.replace(
    'data = {k: v for k, v in data.items() if k in include_fields}',
    'data = {k: v for k, v in data.items() if (not include_fields or k in include_fields)}'
)
open(path, 'w').write(fixed)
print('Fixed flow.py')
"
```

### Dataset

```bash
mkdir data && cd data
curl -L -o DoH-combined.zip "https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/88092/1/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD.zip"
unzip DoH-combined.zip
```

---

## How to Run

### Step 1: Train the detectors

```bash
git checkout detectors
python detector.py --l2 data/l2-total-add.csv --l3 data/l3-total-add.csv --output ./results_full --no_nn
git checkout adversarial
```

### Step 2: White-box attack (upper bound)

Default (interpolated benign targets toward CIRA statistics):

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 20
```

Full pipeline: **correlated tabular sampling**, **tabular correlation report**, **marginal realism gate**, and **`--validate`** CIC-vs-benign table (first CIC-extracted flow per strategy; needs `data/l2-total-add.csv`):

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 20 --target-sampling correlated --correlation-validation --realism-filter --validate
```

(On bash you can split the line with `\` at end of each line; PowerShell: use the single line above.)

Useful flags:

- `--correlation-val-n 3000` — synthetic rows used for the correlation matrix comparison (default 3000).
- `--correlation-report path.json` — where to write tabular correlation metrics.
- `--realism-pass all|any` — require both band and z checks (`all`) or either (`any`, default; better when mapping CIC features to CIRA envelopes).
- `--max-p99-violations`, `--max-z-violations`, `--realism-q-low`, `--realism-q-high`, `--realism-retries` — tune or loosen the marginal gate.

### Step 3: Black-box attack (realistic attacker)

```bash
python blackbox_adversarial_pipeline.py --results ./results_full --flows 20
```

### Step 4: Validate feature matching

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 3 --validate
```

### Step 5: Quick demo (no models needed)

```bash
python main.py --demo
python main.py --doh-demo
```

---

## Files

| File | Purpose |
|---|---|
| `real_adversarial_pipeline.py` | White-box pipeline: Scapy + CICFlowMeter + scaler + optional realism / correlation options |
| `benign_target_sampling.py` | Interpolated and correlated (Ledoit–Wolf MVN) benign CIRA samplers; tabular correlation validation helpers |
| `blackbox_adversarial_pipeline.py` | Realistic attacker: no scaler, binary feedback only |
| `integrate_detectors.py` | Fast synthetic pipeline for quick iteration |
| `traffic_shaper.py` | Core evasion engine: 6 strategies |
| `cira_cic_analyzer.py` | Extracts benign distributions from CIRA-CIC |
| `doh_c2_client.py` | DoH C2 protocol — encodes, encrypts, transmits |
| `adversarial_loop.py` | Standalone attack-detect-adapt loop |
| `main.py` | Entry point for standalone runs |
| `benign_fingerprint.json` | Synthetic benign distributions from literature |
| `synthetic_realistic_doh_generator.ipynb` | Notebook: kNN interpolation for CIRA-like rows (tabular only) |
| `correlation_validation_report.json` | Generated: tabular correlation metrics when `--correlation-validation` is used |
| `real_adversarial_results.csv` | Generated: white-box run summary per strategy and model |
| `blackbox_results.csv` | Black-box results: realistic attacker |
| `adversarial_report.json` | Stub detector baseline results |

---

## Presentation Narrative

1. **The setup:** Flow-based ML detectors trained on 374k real DoH flows. Our job: fool them.
2. **The pipeline:** Traffic shaper → (optional) **correlated or interpolated CIRA benign targets** → Scapy TCP sessions → CICFlowMeter → Classifier; optional **tabular correlation check** and **marginal realism gate** before trusting synthetic evaluation.
3. **SHAP-guided targeting:** PacketLengthMode is the #1 feature. Real benign DoH = tiny packets (~74B). We match that; joint sampling (kNN blend or MVN with shrunk covariance) avoids changing one feature group in isolation.
4. **White-box results:** With classic settings, often near **100% evasion** (upper bound with scaler access). With **strict realism filtering**, reported evasion can drop because fewer synthetic flows pass the CIRA marginal envelope after CIC mapping; interpret those numbers as “evasion among flows that pass our realism checks,” not the raw upper bound.
5. **Black-box results:** 0-50% evasion without model access, RF and XGB are typically robust, GB is partially vulnerable.
6. **The key finding:** The scaler remains a critical defense artifact for white-box comparison. Model behavior can still differ by learner (RF vs GB vs XGB) on out-of-distribution synthetic features.
7. **What this means for defenders:** Flow-based detection is viable if model artifacts are protected; also validate detectors on **distribution-shifted** synthetic traffic, not only scalar SHAP ablations.

---

## Connecting to the Detectors Branch

```bash
# Detectors team runs:
python detector.py --l2 data/l2-total-add.csv --output ./results_full --no_nn

# Adversarial runs both:
python real_adversarial_pipeline.py --results ./results_full --flows 20
python blackbox_adversarial_pipeline.py --results ./results_full --flows 20

# Optional: white-box with benign correlation modeling + tabular validation report
python real_adversarial_pipeline.py --results ./results_full --flows 20 --target-sampling correlated --correlation-validation
```

---

## Realistic Pipeline Results (Mark's KNN Branch)

When using KNN-interpolated benign targets from real CIRA-CIC data instead of independently sampled features, results change for RF:

| Strategy | RF | GB | XGB |
|---|---|---|---|
| naive | 35% | 100% | 100% |
| timing_only | 20% | 100% | 100% |
| size_mimicry | 15% | 100% | 100% |
| cover_traffic | 30% | 100% | 100% |
| full_mimicry | 10% | 100% | 100% |
| adaptive | 40% | 100% | 100% |

**Key finding:** RF is sensitive to statistical realism of generated flows. GB and XGB remain fully evadable regardless.

Results saved in `real_adversarial_results.csv` on the `realistic` branch.

---

## Real C2 Generalization — Raffi's Sliver PCAPs

Evaluated our trained detectors against 5 real Sliver C2 PCAPs (tool never seen in training):

| PCAP | True Label | RF | GB | XGB |
|---|---|---|---|---|
| 01_quiet_beacon | Malicious | 100% | 100% | 100% |
| 02_discovery | Malicious | 100% | 3% | 4% |
| 03_exfiltration | Malicious | 100% | 99% | 99% |
| 04_benign_web | Benign | 0% | 0% | 26% |
| 05_lateral_scan | Malicious | 100% | 4% | 6% |

**Key findings:**
- RF catches all malicious traffic but has false positives on benign web browsing
- GB and XGB miss discovery and lateral scan (only 3-6% detection)
- Models generalize well to C2 behaviors similar to training data, but struggle with unseen patterns
- All PCAPs and flow CSVs saved on the `adversarial` branch

Run evaluation: `python evaluate_real_c2.py`
