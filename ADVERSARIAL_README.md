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
[Scapy] - Full TCP sessions with handshake + data + teardown
      |
PCAP file
      |
[CICFlowMeter] - 28 real flow features (same tool as training data)
      |
[White-box] scaler.transform() + predict_proba()    -> 100% evasion
[Black-box] raw unscaled features + model.predict() -> 0-50% evasion
```

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

```bash
pip install numpy pandas scipy scikit-learn joblib scapy cicflowmeter
```

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

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 20
```

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
| `real_adversarial_pipeline.py` | White-box pipeline: Scapy + CICFlowMeter + scaler |
| `blackbox_adversarial_pipeline.py` | Realistic attacker: no scaler, binary feedback only |
| `integrate_detectors.py` | Fast synthetic pipeline for quick iteration |
| `traffic_shaper.py` | Core evasion engine: 6 strategies |
| `cira_cic_analyzer.py` | Extracts benign distributions from CIRA-CIC |
| `doh_c2_client.py` | DoH C2 protocol — encodes, encrypts, transmits |
| `adversarial_loop.py` | Standalone attack-detect-adapt loop |
| `main.py` | Entry point for standalone runs |
| `benign_fingerprint.json` | Synthetic benign distributions from literature |
| `real_adversarial_results.csv` | White-box results: 20 flows per strategy |
| `blackbox_results.csv` | Black-box results: realistic attacker |
| `adversarial_report.json` | Stub detector baseline results |

---

## Presentation Narrative

1. **The setup:** Flow-based ML detectors trained on 374k real DoH flows. Our job: fool them.
2. **The pipeline:** Traffic shaper → Scapy TCP sessions → CICFlowMeter → Classifier.
3. **SHAP-guided targeting:** PacketLengthMode is the #1 feature. Real benign DoH = tiny packets (~74B). We match that.
4. **White-box results:** 100% evasion across all 6 strategies and all 3 models, representing the upper bound assuming model theft.
5. **Black-box results:** 0-50% evasion without model access, RF and XGB are robust, GB is partially vulnerable.
6. **The key finding:** The scaler is the critical defense artifact. Keeping it private makes RF and XGB nearly impenetrable, but stealing it makes evasion trivial.
7. **What this means for defenders:** Flow-based detection is viable IF model artifacts are kept strictly private. The architecture is sound — operational security is the weak point.

---

## Connecting to the Detectors Branch

```bash
# Detectors team runs:
python detector.py --l2 data/l2-total-add.csv --output ./results_full --no_nn

# Adversarial runs both:
python real_adversarial_pipeline.py --results ./results_full --flows 20
python blackbox_adversarial_pipeline.py --results ./results_full --flows 20
```
