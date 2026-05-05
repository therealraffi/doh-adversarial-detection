# doh-adversarial-detection

ML classifiers that distinguish benign DoH traffic from malicious DoH tunnel traffic, trained on the Hokkaido University combined dataset. Includes Random Forest, Gradient Boosting, XGBoost, and a PyTorch MLP, with SHAP explainability and per-tool detection breakdowns.

---

## Project Structure

```
doh-adversarial-detection/
├── detector.py          # Main training + evaluation script
├── requirements.txt     # Python dependencies
├── netenv/              # Virtualenv (pre-built, do not re-create)
├── data/
│   ├── l2-total-add.csv # Layer 2: Benign vs. Malicious (374,803 rows)
│   ├── l3-total-add.csv # Layer 3: per-tool tunnel labels (354,996 rows)
│   ├── l1-total-add.csv # Layer 1: Non-DoH vs. DoH (not used by detector.py)
│   └── README.txt       # Dataset citation and license
└── results/             # Output artifacts (created on first run)
    ├── *.joblib / mlp.pt           # Serialized models
    ├── scaler.joblib               # Fitted StandardScaler
    ├── feature_limits.npy          # (28,2) malicious feature percentiles
    ├── pearson_matrix.npy          # (28,28) feature correlation matrix
    ├── feature_names.json          # Ordered feature column names
    ├── results.txt                 # Full evaluation report
    ├── per_tool_results.txt        # F1 breakdown by tunnel tool
    ├── *_confusion_matrix.png      # Per-model confusion matrices
    ├── *_roc_curve.png             # Per-model ROC curves
    └── *_shap_importance.png/json  # Per-model SHAP feature importance
```

---

## Environment Setup

This project runs on UVA's HPC cluster (Rivanna/Afton). The virtualenv at `netenv/` already has all dependencies installed.

### 1. Get an interactive GPU node

```bash
ijob -c 1 -A ml_at_uva -p gpu --gres=gpu:1 --time=0-01:00:00 --mem=64G
```

### 2. Load required modules

```bash
ml cuda miniforge gcc
```

### 3. Activate the virtualenv

```bash
source /scratch/rhm4nj/netsec/doh-adversarial-detection/netenv/bin/activate
```

You should see `(netenv)` prepended to your prompt.

### Rebuilding the virtualenv from scratch (if needed)

```bash
python -m venv netenv
source netenv/bin/activate
pip install -r requirements.txt
```

---

## Data
```bash
mkdir data
cd data
wget -O DoH-combined.zip "https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/88092/1/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD.zip"
unzip DoH-combined.zip
```

## Usage

### Full run (all four models, both CSVs)

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results \
    --seed 42
```

### Fast run — skip the neural network

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results \
    --no_nn
```

### Development / smoke test — subsample rows

```bash
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_test \
    --sample 5000 \
    --no_nn
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--l2` | *(required)* | Path to `l2-total-add.csv` |
| `--l3` | *(optional)* | Path to `l3-total-add.csv` for per-tool breakdown |
| `--output` | `./results` | Directory to write all output artifacts |
| `--seed` | `42` | Random seed for reproducibility |
| `--test_size` | `0.2` | Fraction of data held out for testing |
| `--no_nn` | off | Skip MLP training (faster runs without GPU) |
| `--sample` | off | Randomly subsample N rows from l2 (for quick iteration) |

---

## Results (full run)

All four models achieve near-perfect detection on this dataset:

| Model | Accuracy | F1 (binary) | ROC-AUC |
|---|---|---|---|
| Random Forest | ~0.999 | ~0.999 | ~1.000 |
| Gradient Boosting | ~0.999 | ~0.999 | ~1.000 |
| XGBoost | ~0.999 | ~0.999 | ~1.000 |
| MLP | ~0.999 | ~0.999 | ~1.000 |

### Per-tool detection F1 (full dataset)

```
Tool            | RF_F1   | GB_F1   | XGB_F1  | MLP_F1
--------------------------------------------------------
dns2tcp         | 1.0000  | 1.0000  | 1.0000  | 0.9997
dnscat2         | 1.0000  | 1.0000  | 1.0000  | 0.9993
dnstt           | 1.0000  | 1.0000  | 1.0000  | 1.0000
iodine          | 1.0000  | 1.0000  | 0.9999  | 0.9987
tcp-over-dns    | 1.0000  | 1.0000  | 1.0000  | 1.0000
tuns            | 1.0000  | 1.0000  | 1.0000  | 1.0000
```

---

## Dataset

The Hokkaido University combined dataset merges:

1. **CIRA-CIC-DoHBrw-2020** — DoH browser traffic captures  
2. **DoH-Tunnel-Traffic-HKD** — six tunnel tool implementations

If you use this dataset, cite both papers:

> M. MontazeriShatoori et al., "Detection of DoH Tunnels using Time-series Classification of Encrypted Traffic," IEEE CyberSciTech, 2020. https://ieeexplore.ieee.org/document/9251211

> R. Mitsuhashi et al., "Malicious DNS Tunnel Tool Recognition using Persistent DoH Traffic Analysis," IEEE TNSM, 2022. https://ieeexplore.ieee.org/document/9924534

Dataset questions: mitsuhashi@os.ecc.u-tokyo.ac.jp

---

## c2-lab

A Dockerized environment for simulating DNS/DoH-based C2 traffic using [Sliver](https://github.com/BishopFox/sliver). Used to generate the labeled packet captures that feed the detectors above.

### Prerequisites

- Docker + Docker Compose
- `tcpdump` on the host
- `sliver-client` installed locally

### Network Layout

| Container  | IP          | Role                        |
| ---------- | ----------- | --------------------------- |
| `resolver` | 172.20.0.10 | CoreDNS (DoT/DoH forwarder) |
| `attacker` | 172.20.0.20 | Sliver C2 server            |
| `victim`   | 172.20.0.30 | Ubuntu host running implant |

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
