# DoHShield: DoH Tunnel Detection and Adversarial Evaluation

This repository contains a flow-based machine learning framework for detecting DNS-over-HTTPS (DoH) tunnel and C2 traffic, along with adversarial pipelines that test whether those detectors can be evaded.

The project uses the Hokkaido University combined dataset, which includes CIRA-CIC-DoHBrw-2020 and DoH-Tunnel-Traffic-HKD. The detector pipeline trains classifiers on flow-level network features, while the adversarial pipelines generate synthetic DoH C2 traffic, extract the same feature set, and test whether the trained models still detect the traffic.

---

## Overview

This project has two connected parts:

1. **Detector training**
   - Trains Random Forest, Gradient Boosting, XGBoost, and optionally a PyTorch MLP.
   - Uses flow-level features from benign and malicious DoH traffic.
   - Saves model artifacts, the fitted scaler, feature names, SHAP explanations, and evaluation reports.

2. **Adversarial evaluation**
   - Generates synthetic DoH C2 traffic using different evasion strategies.
   - Converts shaped traffic into packet/session-level data using Scapy.
   - Extracts flow features using CICFlowMeter.
   - Tests whether the trained detectors classify the generated traffic as malicious or benign.

This means users can now train and test everything from one branch.

---

## Project Structure

```text
doh-adversarial-detection/
├── detector.py                        # Main detector training and evaluation script
├── real_adversarial_pipeline.py       # White-box adversarial evaluation pipeline
├── blackbox_adversarial_pipeline.py   # Black-box adversarial evaluation pipeline
├── benign_target_sampling.py          # Benign target sampling and correlation validation
├── traffic_shaper.py                  # Core traffic shaping and evasion strategies
├── cira_cic_analyzer.py               # Extracts benign distributions from CIRA-CIC data
├── integrate_detectors.py             # Fast synthetic feature-level attack baseline
├── doh_c2_client.py                   # DoH C2 protocol logic
├── adversarial_loop.py                # Standalone attack-detect-adapt loop
├── main.py                            # Entry point for quick demos
├── requirements.txt                   # Python dependencies
├── c2-lab/                            # Docker lab for generating labeled C2/benign captures
│   ├── docker-compose.yml
│   └── coredns/                       # CoreDNS config and TLS certificates
├── data/                              # Dataset CSVs (gitignored; see Dataset section)
│   ├── l2-total-add.csv               # Layer 2: Benign vs. Malicious DoH
│   ├── l3-total-add.csv               # Layer 3: malicious tunnel tool labels
│   ├── l1-total-add.csv               # Layer 1: Non-DoH vs. DoH, not used by detector.py
│   └── README.txt                     # Dataset citation and license
└── results_full/                      # Output directory created after training/runs
    ├── rf.joblib                      # Trained Random Forest model
    ├── gb.joblib                      # Trained Gradient Boosting model
    ├── xgb.joblib                     # Trained XGBoost model
    ├── mlp.pt                         # Optional trained PyTorch MLP
    ├── scaler.joblib                  # Fitted StandardScaler
    ├── feature_names.json             # Ordered detector feature names
    ├── feature_limits.npy             # Feature percentile limits
    ├── pearson_matrix.npy             # Feature correlation matrix
    ├── results.txt                    # Detector evaluation report
    ├── per_tool_results.txt           # Per-tool detection breakdown
    ├── *_confusion_matrix.png         # Confusion matrices
    ├── *_roc_curve.png                # ROC curves
    ├── *_shap_importance.png/json     # SHAP feature importance
    ├── real_adversarial_results.csv   # White-box attack results
    └── blackbox_results.csv           # Black-box attack results
```

---

## Environment Setup

This project can be run locally or on UVA's HPC cluster.

### Option 1: Local Setup

```bash
python -m venv netenv
source netenv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv netenv
.\netenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: UVA Rivanna/Afton Setup

Get an interactive GPU node:

```bash
ijob -c 1 -A ml_at_uva -p gpu --gres=gpu:1 --time=0-01:00:00 --mem=64G
```

Load modules:

```bash
ml cuda miniforge gcc
```

Activate your environment:

```bash
source netenv/bin/activate
```

If the environment does not already exist:

```bash
python -m venv netenv
source netenv/bin/activate
pip install -r requirements.txt
```

---

## Dependencies

The main dependencies are:

```bash
pip install numpy pandas scipy scikit-learn joblib xgboost scapy cicflowmeter matplotlib seaborn shap torch
```

Notes:

- `xgboost` is required to train and load `xgb.joblib`.
- `torch` is only needed if running the MLP. Use `--no_nn` to skip neural network training.
- `pyarrow` or `fastparquet` is optional, but needed if using parquet reference files.
- `cicflowmeter` is required for the packet-to-flow feature extraction used by the adversarial pipelines.

---

## Fix CICFlowMeter v0.5.0 Bugs

If CICFlowMeter fails during extraction, apply these two patches:

```bash
python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/sniffer.py'
txt = open(path).read()
fixed = txt.replace('if fields is not None:', 'if fields is not None and fields is not False and fields != True:')
open(path, 'w').write(fixed)
print('Fixed sniffer.py')
"
```

```bash
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

Update the path if your Python environment is not located at `/opt/anaconda3/lib/python3.12/`.

---

## Dataset

Download and unzip the Hokkaido University combined DoH dataset:

```bash
mkdir data
cd data
wget -O DoH-combined.zip "https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/88092/1/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD.zip"
unzip DoH-combined.zip
cd ..
```

The main files used by this project are:

```text
data/l2-total-add.csv
data/l3-total-add.csv
```

`l2-total-add.csv` is used for binary benign vs. malicious DoH detection.

`l3-total-add.csv` is used for the per-tool malicious tunnel breakdown.

---

## How to Run

### Step 1: Train the Detectors

Fast run without the neural network:

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_full \
    --no_nn
```

Full run with all four models:

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_full \
    --seed 42
```

Development smoke test with a smaller sample:

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_test \
    --sample 5000 \
    --no_nn
```

The detector step must be run before the adversarial pipelines because it creates the model files, scaler, and feature names used during attack evaluation.

---

### Step 2: Run the White-box Adversarial Pipeline

The white-box pipeline assumes the attacker has access to the trained models, scaler, feature names, and model feedback. This is an upper-bound attack setting.

Default run using interpolated benign targets:

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 20
```

Full run with correlated tabular sampling, tabular correlation validation, marginal realism filtering, and CIC-vs-benign validation:

```bash
python real_adversarial_pipeline.py \
    --results ./results_full \
    --flows 20 \
    --target-sampling correlated \
    --correlation-validation \
    --realism-filter \
    --validate
```

Useful realism and validation flags:

```bash
python real_adversarial_pipeline.py \
    --results ./results_full \
    --flows 20 \
    --target-sampling correlated \
    --correlation-validation \
    --correlation-val-n 3000 \
    --correlation-report correlation_validation_report.json \
    --realism-filter \
    --realism-pass any \
    --validate
```

On PowerShell, use the same command on one line instead of using `\` line breaks.

---

### Step 3: Run the Black-box Adversarial Pipeline

The black-box pipeline represents a more realistic attacker. The attacker can observe benign traffic and use public tools, but does not have access to the scaler or model internals.

```bash
python blackbox_adversarial_pipeline.py --results ./results_full --flows 20
```

---

### Step 4: Validate Feature Matching

To print a short CICFlowMeter feature comparison against benign CIRA traffic:

```bash
python real_adversarial_pipeline.py --results ./results_full --flows 3 --validate
```

---

### Step 5: Quick Demo Without Trained Models

```bash
python main.py --demo
python main.py --doh-demo
```

---

## Detector CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--l2` | required | Path to `l2-total-add.csv` |
| `--l3` | optional | Path to `l3-total-add.csv` for per-tool breakdown |
| `--output` | `./results` | Directory to write model artifacts and results |
| `--seed` | `42` | Random seed for reproducibility |
| `--test_size` | `0.2` | Fraction of data held out for testing |
| `--no_nn` | off | Skip MLP training |
| `--sample` | off | Randomly subsample N rows from l2 for quick testing |

---

## White-box Pipeline Flags

| Flag | Description |
|---|---|
| `--results` | Directory containing trained model artifacts from `detector.py`. |
| `--flows` | Number of generated flows per evasion strategy. |
| `--target-sampling interpolated` | Default. kNN-style blend of nearby benign CIRA rows with noise and quantile clipping. |
| `--target-sampling correlated` | Multivariate Gaussian sampling in standardized space using Ledoit-Wolf shrunk covariance. |
| `--target-sampling legacy` | Original random wire-stat sampling without CIRA row targeting. |
| `--reference-data` | Optional path to the benign reference dataset. If omitted, the pipeline tries to use `L2-BenignDoH-MaliciousDoH.parquet` or `data/l2-total-add.csv`. |
| `--correlation-validation` | Runs tabular correlation validation between real benign CIRA rows and synthetic sampled CIRA-style rows. |
| `--correlation-val-n` | Number of synthetic rows used for correlation validation. Default is usually 3000. |
| `--correlation-report` | Path to write the tabular correlation validation report, usually `correlation_validation_report.json`. |
| `--realism-filter` | Applies marginal realism checks after CICFlowMeter extraction and feature mapping. |
| `--realism-pass all\|any` | Controls whether all realism checks must pass or whether any check passing is enough. `any` is usually better for cross-domain tolerance. |
| `--max-p99-violations` | Tunes how many upper-quantile violations are allowed during marginal realism filtering. |
| `--max-z-violations` | Tunes how many z-score violations are allowed during marginal realism filtering. |
| `--realism-q-low` | Lower quantile bound for the realism filter. |
| `--realism-q-high` | Upper quantile bound for the realism filter. |
| `--realism-retries` | Number of retries allowed when generated flows fail the realism filter. |
| `--validate` | Prints a CIC column vs. benign comparison for the first extracted flow in each strategy. |

---

## Architecture

### Detector Training

```text
CIRA/HKD CSVs
      |
Feature cleaning + preprocessing
      |
Train/test split
      |
StandardScaler
      |
RF / GB / XGB / optional MLP
      |
Saved model artifacts in results_full/
```

### Adversarial Evaluation

```text
C2 payload
      |
traffic_shaper.py
      |
Optional benign target sampling
      |  interpolated, correlated, or legacy
      |
Scapy TCP sessions
      |
Temporary PCAP
      |
CICFlowMeter feature extraction
      |
Feature mapping to detector columns
      |
Optional realism filter / validation
      |
White-box: scaler.transform() + model.predict_proba()
Black-box: raw unscaled features + model.predict()
      |
Detected or evaded
```

---

## Evasion Strategies

| Strategy | Description |
|---|---|
| `naive` | Fixed 5 second beacon with no shaping |
| `timing_only` | Randomized inter-arrival times matching browser-like traffic |
| `size_mimicry` | Timing plus packet sizes matching benign DoH distributions |
| `cover_traffic` | Adds decoy queries to real resolvers such as Google or Cloudflare |
| `full_mimicry` | Combines timing, size, cover traffic, and burst grouping |
| `adaptive` | Full mimicry with a classifier feedback loop |

---

## Realistic Benign Targeting and Validation

The white-box pipeline can draw CIRA L2-style benign targets from the reference dataset, map those targets into Scapy session parameters, generate packet-level traffic, and then run CICFlowMeter on the resulting synthetic PCAPs.

This is important because the generated traffic does not directly reproduce raw CIRA rows. The pipeline goes through several transformations:

```text
CIRA benign target row
      |
traffic_params_from_cira_features()
      |
Scapy TCP session generation
      |
CICFlowMeter extraction
      |
map_features()
      |
Detector feature vector
```

Because CICFlowMeter features and CIRA feature columns are not always identical, validation is split into multiple checks instead of assuming exact row-level reproduction.

### Validation Layers

| Layer | What it checks |
|---|---|
| `--correlation-validation` | Compares Pearson correlation structure from real benign CIRA rows against synthetic CIRA-style rows from the active sampler. This writes `correlation_validation_report.json`, including Frobenius norm of the matrix difference, max entry error, and mean off-diagonal error. This check happens in CIRA feature space only. |
| `--realism-filter` | After CICFlowMeter extraction and `map_features()`, compares detector features against benign quantile bands and z-score limits from the reference table. This is a marginal realism check after packet generation. |
| `--validate` | Prints a short CIC column vs. benign comparison for the first successful CIC extraction on flow index 0 within each strategy. This does not wait for `--realism-filter` acceptance. It uses `data/l2-total-add.csv`, so if that file is missing, validation prints nothing. |

### Target Sampling Modes

| Mode | Behavior |
|---|---|
| `interpolated` | Default mode. Uses a kNN-style blend of nearby benign rows, adds small noise, and clips to benign quantiles. This follows the same general idea as `synthetic_realistic_doh_generator.ipynb`. |
| `correlated` | Uses a multivariate Gaussian in standardized space with Ledoit-Wolf shrunk covariance, then inverse-transforms and clips values. This explicitly models linear correlations among CIRA columns. |
| `legacy` | Uses the older random wire-stat generation method without sampling CIRA target rows. |

Reference data auto-resolves to `L2-BenignDoH-MaliciousDoH.parquet` or `data/l2-total-add.csv` when present. You can override this with `--reference-data`.

### Environment Variables

| Variable | Purpose |
|---|---|
| `DOH_CFM_FORCE_CLI=1` | Forces CICFlowMeter CLI instead of in-process extraction. |
| `DOH_CFM_DEBUG=1` | Prints CICFlowMeter failures when falling back. |

### Important Interpretation Note

The realism checks are sanity checks, not proof that a generated flow is identical to a real benign flow. The pipeline maps from CIRA-style target features to Scapy packets and then back into CICFlowMeter features, so some mismatch is expected. For that reason, `--realism-pass any` is often more useful than `--realism-pass all` when checking cross-domain feature realism.

---

## Results Summary

### Detector Results

All four detector models achieve near-perfect offline performance on the original dataset.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | ~0.999 | ~0.999 | ~1.000 |
| Gradient Boosting | ~0.999 | ~0.999 | ~1.000 |
| XGBoost | ~0.999 | ~0.999 | ~1.000 |
| MLP | ~0.999 | ~0.999 | ~1.000 |

Per-tool detection is also near-perfect across the six malicious tunnel tools:

```text
Tool            | RF_F1   | GB_F1   | XGB_F1  | MLP_F1
--------------------------------------------------------
dns2tcp         | 1.0000  | 1.0000  | 1.0000  | 0.9997
dnscat2         | 1.0000  | 1.0000  | 1.0000  | 0.9993
dnstt           | 1.0000  | 1.0000  | 1.0000  | 1.0000
iodine          | 1.0000  | 1.0000  | 0.9999  | 0.9987
tcp-over-dns    | 1.0000  | 1.0000  | 1.0000  | 1.0000
tuns            | 1.0000  | 1.0000  | 1.0000  | 1.0000
```

### White-box Attack Results

In the white-box setting, the attacker has access to the scaler, feature names, model internals, and model feedback.

| Strategy | RF | GB | XGB |
|---|---|---|---|
| `naive` | 100% | 100% | 100% |
| `timing_only` | 100% | 100% | 100% |
| `size_mimicry` | 100% | 100% | 100% |
| `cover_traffic` | 100% | 100% | 100% |
| `full_mimicry` | 100% | 100% | 100% |
| `adaptive` | 100% | 100% | 100% |

### Black-box Attack Results

In the black-box setting, the attacker has observed benign traffic and public tools, but does not have the scaler or model internals.

| Strategy | RF | GB | XGB |
|---|---|---|---|
| `naive` | 0% | 0% | 0% |
| `timing_only` | 0% | 40% | 0% |
| `size_mimicry` | 0% | 40% | 0% |
| `cover_traffic` | 0% | 25% | 0% |
| `full_mimicry` | 0% | 50% | 0% |
| `adaptive` | 0% | 50% | 0% |

---

## Key Findings

1. Offline detector accuracy is not the same as deployment robustness.
2. Attacker knowledge level is the dominant factor in evasion success.
3. The fitted scaler is a critical defense artifact. With scaler access, evasion becomes much easier.
4. Random Forest and XGBoost are more robust in the black-box setting.
5. Gradient Boosting is fastest, but more vulnerable to black-box evasion.
6. Synthetic feature edits should be validated because changing one feature group in isolation can create unrealistic traffic.
7. Correlation-aware benign sampling helps avoid treating features as independent.
8. The realism filter gives a stricter view of attack quality by checking whether generated flows stay within benign marginal ranges after packet generation and CICFlowMeter extraction.
9. With strict realism filtering, evasion results may drop because fewer generated flows pass the realism gate. Those results should be interpreted as evasion among flows that pass the realism checks, not necessarily the raw upper-bound evasion rate.

---

## c2-lab

A Dockerized environment for simulating DNS/DoH-based C2 traffic using [Sliver](https://github.com/BishopFox/sliver). Used to generate labeled packet captures for detector training and evaluation.

### Prerequisites

- Docker + Docker Compose
- `tcpdump` on the host
- `sliver-client` installed locally

### Network Layout

| Container  | IP            | Role                        |
| ---------- | ------------- | --------------------------- |
| `resolver` | 172.20.0.10   | CoreDNS (DoT/DoH forwarder) |
| `attacker` | 172.20.0.20   | Sliver C2 server            |
| `victim`   | 172.20.0.30   | Ubuntu host running implant |

### Setup

```bash
cd c2-lab
docker compose up -d
```

Connect the Sliver client to the attacker container:

```bash
docker exec -it c2-lab-attacker-1 sliver-server
```

If the Sliver server isn't running yet, start it inside the container first, then connect with `sliver-client` from your host.

### Running a Capture Session

1. **Find the project bridge** (run on host):

   ```bash
   docker network inspect c2-lab_c2net --format '{{.Id}}' | cut -c1-12
   # bridge will be br-<id>
   ```

2. **Start recording** on host (replace `br-XXXX` with your bridge):

   ```bash
   tcpdump -i br-XXXX -w captures/<scenario_name>.pcap
   ```

3. **Start the DNS listener** inside Sliver:

   ```
   dns --domains lab.local.
   ```

4. **Force victim DNS to attacker** (if resolver is bypassed):

   ```bash
   docker exec c2-lab-victim-1 bash -c \
     "echo 'nameserver 172.20.0.20' > /etc/resolv.conf && \
      echo '172.20.0.20 lab.local' >> /etc/hosts"
   ```

5. **Run the implant** on the victim:

   ```bash
   docker exec c2-lab-victim-1 /implants/doh_implant
   ```

6. Perform the scenario interactions, then kill the implant and stop `tcpdump`.

### Capture Scenarios

| File                   | Label  | Description                    |
| ---------------------- | ------ | ------------------------------ |
| `01_quiet_beacon.pcap` | C2     | Idle beacon only (~10 min)     |
| `02_discovery.pcap`    | C2     | `ls -R`, `ps`, `whoami`        |
| `03_exfiltration.pcap` | C2     | `download` of a large file     |
| `04_benign_web.pcap`   | Benign | `curl` loop to legitimate URLs |
| `05_lateral_scan.pcap` | C2     | Internal `ping`/`netstat`      |

### Teardown

```bash
docker compose down
```

---

## Dataset Citation

The Hokkaido University combined dataset merges:

1. **CIRA-CIC-DoHBrw-2020**, which contains DoH browser traffic captures.
2. **DoH-Tunnel-Traffic-HKD**, which contains six malicious DoH tunnel tool implementations.

If you use this dataset, cite both papers:

> M. MontazeriShatoori et al., "Detection of DoH Tunnels using Time-series Classification of Encrypted Traffic," IEEE CyberSciTech, 2020. https://ieeexplore.ieee.org/document/9251211

> R. Mitsuhashi et al., "Malicious DNS Tunnel Tool Recognition using Persistent DoH Traffic Analysis," IEEE TNSM, 2022. https://ieeexplore.ieee.org/document/9924534

Dataset questions: mitsuhashi@os.ecc.u-tokyo.ac.jp
