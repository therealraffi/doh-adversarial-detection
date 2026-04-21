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
