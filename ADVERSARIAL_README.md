# Adversarial DoH C2 Evasion Framework

**Branch:** `adversarial` | **Role:** Attacker  
**Project:** DoH Tunnel / C2 Detection — Network Security Final Project  
**Dataset:** Hokkaido University Combined Dataset (CIRA-CIC-DoHBrw-2020 + DoH-Tunnel-Traffic-HKD)

---

## Overview

This branch implements the **adversarial side** of the project. The goal is to generate DoH C2 traffic that evades the flow-based ML classifiers built in the `detectors` branch (RF, GB, XGB, MLP).

We built two pipelines:

1. **`integrate_detectors.py`** — Fast synthetic pipeline. Computes flow features directly from simulated schedules without generating real packets. Good for quick iteration but features don't perfectly match training data.

2. **`real_adversarial_pipeline.py`** — Real pipeline. Converts evasion schedules into actual TCP packets via Scapy, extracts features using CICFlowMeter (same tool the training data used), then scores against the detectors. This is the accurate, defensible result.

---

## Key Findings

### White-box Attack (used actual dataset distributions)
| Strategy | GB | RF | XGB |
|---|---|---|---|
| naive | 100% | 100% | 100% |
| timing_only | 100% | 95% | 100% |
| size_mimicry | 100% | 100% | 100% |
| cover_traffic | 100% | 100% | 100% |
| full_mimicry | 100% | 100% | 100% |
| adaptive | 100% | 95% | 100% |

### Gray-box Attack (used synthetic fingerprint from published literature — independent of training data)
| Strategy | GB | RF | XGB |
|---|---|---|---|
| naive | 100% | 100% | 100% |
| timing_only | 100% | 100% | 100% |
| size_mimicry | 100% | 100% | 100% |
| cover_traffic | 100% | 100% | 100% |
| full_mimicry | 100% | 100% | 100% |
| adaptive | 100% | 90% | 100% |

**The attack works with 90-100% evasion regardless of whether the attacker has access to the training data.**

---

## Why It Works — SHAP Analysis

We ran SHAP on all three models to identify which features drive detection. The top features were:

| Feature | RF Importance | XGB Importance |
|---|---|---|
| PacketLengthMode | 0.124 | 5.624 |
| PacketLengthMean | 0.037 | 1.012 |
| FlowBytesReceived | 0.037 | 1.518 |
| PacketTimeStandardDeviation | 0.040 | 0.096 |

The key insight: **real benign DoH traffic has very small packets** (mode ~74B, mean ~137B). Naive C2 tools send large, uniform packets — trivially detectable. Our attack generates flows matching the real benign distributions.

Real benign distributions (from `l2-total-add.csv`):
- `PacketLengthMode`: mean=74B, std=20B
- `PacketLengthMean`: mean=137B, std=82B  
- `FlowBytesSent`: mean=10KB (requires ~400-600 packets per flow)
- `PacketTimeStd`: mean=11ms (timing in milliseconds, not seconds)

Two bugs we had to fix to get real results:
1. CICFlowMeter outputs time in **seconds**, but the training data used **milliseconds** — multiply all IAT features by 1000
2. Our flows had too few packets (~50) — benign flows have 400-600 packets to reach ~62KB `FlowBytesSent`

---

## Architecture

```
traffic_shaper.py          → Generates query schedules (6 evasion strategies)
        ↓
real_adversarial_pipeline.py
        ↓
  [Scapy] → TCP packets with realistic sizes/timing
        ↓
  [CICFlowMeter] → 28 real flow features
        ↓
  [RF / GB / XGB] → Classification score
        ↓
  real_adversarial_results.csv
```

---

## Evasion Strategies

| Strategy | Description | How it works |
|---|---|---|
| `naive` | Fixed 5s beacon, no shaping | Baseline — trivially caught without size/timing fixes |
| `timing_only` | Randomized IAT matching browser distribution | Fixes timing but not size |
| `size_mimicry` | Timing + EDNS0 padding to match packet sizes | Adds size shaping |
| `cover_traffic` | Interleaves decoy queries to real resolvers | Mixes C2 with benign-looking queries |
| `full_mimicry` | Everything + burst grouping mimicking page loads | Full statistical mimicry |
| `adaptive` | Full mimicry + classifier feedback loop | Tightens parameters based on detection results |

With correct feature generation, **all 6 strategies achieve 90-100% evasion**.

---

## Setup

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn joblib scapy cicflowmeter
```

### Fix CICFlowMeter Bug (v0.5.0 has two bugs — patch them)

```bash
# Fix 1: sniffer.py
python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/sniffer.py'
txt = open(path).read()
fixed = txt.replace('if fields is not None:', 'if fields is not None and fields is not False and fields != True:')
open(path, 'w').write(fixed)
print('Fixed sniffer.py')
"

# Fix 2: flow.py
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

The Hokkaido dataset is required to train the detectors:

```bash
mkdir data && cd data
curl -L -o DoH-combined.zip "https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/88092/1/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD.zip"
unzip DoH-combined.zip
```

---

## How to Run

### Step 1 — Train the detectors (from `detectors` branch)

```bash
git checkout detectors
python detector.py --l2 data/l2-total-add.csv --l3 data/l3-total-add.csv --output ./results_full --no_nn
git checkout adversarial
```

### Step 2 — Run the real adversarial pipeline

```bash
# Full run (20 flows per strategy, ~10 mins)
python real_adversarial_pipeline.py --results ./results_full --flows 20

# With feature validation (shows our features vs benign)
python real_adversarial_pipeline.py --results ./results_full --flows 3 --validate

# Gray-box attack (uses synthetic fingerprint, independent of training data)
python real_adversarial_pipeline.py --results ./results_full --fingerprint benign_fingerprint.json --flows 20
```

### Step 3 — Run the fast synthetic pipeline (for quick iteration)

```bash
python integrate_detectors.py --results ./results_full --flows 20
```

### Step 4 — Standalone demo (no models needed)

```bash
python main.py --demo        # Shows traffic shaping across all strategies
python main.py --doh-demo    # Shows DoH C2 encoding dry-run
python main.py --flows 50    # Runs full adversarial loop with stub detector
```

---

## Files

| File | Purpose |
|---|---|
| `real_adversarial_pipeline.py` | **Main file.** Real Scapy+CICFlowMeter pipeline |
| `integrate_detectors.py` | Fast synthetic pipeline — plugs into team's models |
| `traffic_shaper.py` | Core evasion engine — 6 strategies, schedule generation |
| `cira_cic_analyzer.py` | Extracts benign distributions from CIRA-CIC dataset |
| `doh_c2_client.py` | DoH C2 protocol — encodes, encrypts, sends C2 data |
| `adversarial_loop.py` | Standalone attack-detect-adapt feedback loop |
| `main.py` | Entry point for standalone runs |
| `benign_fingerprint.json` | Synthetic benign feature distributions (from literature) |
| `real_adversarial_results.csv` | Final evasion results — 20 flows per strategy |
| `adversarial_report.json` | Results from stub detector baseline |

---

## Threat Model

**White-box attack:** Attacker has access to the training dataset distributions. Used to establish upper bound on evasion (95-100%).

**Gray-box attack:** Attacker only has general knowledge of DoH traffic patterns from published literature (CIRA-CIC papers). Used to show the attack is realistic. Results: 90-100% evasion — nearly identical to white-box.

**What this reveals:** Flow-based statistical detection is fundamentally vulnerable to an attacker who can match benign traffic distributions at the packet level. The classifier cannot distinguish C2 traffic from benign if the flow statistics are identical. The only defenses would be content inspection (which DoH encrypts by design) or longer-window behavioral analysis.

---

## Connecting to the Detectors Branch

Once the detector team trains models and exports them:

```bash
# They run:
python detector.py --l2 data/l2-total-add.csv --output ./results_full --no_nn

# You run:
python real_adversarial_pipeline.py --results ./results_full --flows 20
```

The pipeline automatically loads `rf.joblib`, `gb.joblib`, `xgb.joblib`, and `scaler.joblib` from the results directory.

---

## Presentation Narrative

1. **The attack pipeline:** Evasion shaper → Scapy TCP sessions → CICFlowMeter → Classifier
2. **SHAP-guided targeting:** We identified the top features (PacketLengthMode, FlowBytesSent) and reverse-engineered the benign distributions
3. **Results:** 90-100% evasion across all 6 strategies, all 3 models, both white-box and gray-box settings
4. **The finding:** Flow-based detection alone is insufficient. An attacker who mimics benign traffic statistics at the packet level can fully evade these classifiers.
5. **Limitation:** This is a distribution-matching attack. A real attacker would need to observe benign DoH traffic on their network to estimate these distributions — which is feasible.
