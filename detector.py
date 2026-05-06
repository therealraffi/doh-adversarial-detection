import argparse
import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


TUNNEL_TOOLS = {"dns2tcp", "dnscat2", "iodine", "dnstt", "tcp-over-dns", "tuns"}
META_PATTERNS = ("ip", "port", "timestamp", "duration", "label", "class")


def parse_args():
    p = argparse.ArgumentParser(description="DoH tunnel traffic detector")
    p.add_argument("--l2", required=True, help="Path to l2-total-add.csv")
    p.add_argument("--l3", default=None, help="Path to l3-total-add.csv (optional)")
    # p.add_argument("--output", default="./out")
    p.add_argument("--output", default="./results", help="Output directory")
    # p.add_argument("--seed", type=int, default=0)
    # p.add_argument("--seed", type=int, default=123)
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # p.add_argument("--test_size", type=float, default=0.15)
    # p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--no_nn", action="store_true", help="Skip neural network model")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N rows for fast iteration")
    return p.parse_args()


def find_label_col(df):
    for candidate in ("label", "Label", "class", "Class"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def is_meta_col(col_name):
    lower = col_name.lower().lstrip("﻿")
    return any(pat in lower for pat in META_PATTERNS)


def identify_benign_label(unique_labels):
    nonTools = [v for v in unique_labels if v.lower() not in TUNNEL_TOOLS]
    if len(nonTools) == 1:
        return nonTools[0]
    benignKw = ("benign", "normal", "doh", "safe", "legitimate")
    for v in nonTools:
        if any(kw in v.lower() for kw in benignKw):
            return v
    return nonTools[0] if nonTools else unique_labels[0]


def load_and_inspect(path, sample=None, seed=42):
    print(f"\n{'='*60}")
    print(f"Loading: {path}")
    print(f"{'='*60}")

    # df = pd.read_csv(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿") for c in df.columns]


    if sample is not None:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
        print(f"  [sampled {len(df)} rows]")

    print(f"Shape: {df.shape}")

    label_col = find_label_col(df)
    print(f"\nLabel column: '{label_col}'")
    print(df[label_col].value_counts().to_string())

    unique_labels = df[label_col].unique().tolist()
    benign_label = identify_benign_label(unique_labels)
    print(f"\nBenign label identified as: '{benign_label}'")

    y = (df[label_col] != benign_label).astype(int)
    print("\nBinarized class counts:")
    print(y.value_counts().rename({0: "0 (benign)", 1: "1 (malicious)"}).to_string())

    feature_cols = [c for c in df.columns if not is_meta_col(c)]
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols)
    assert len(feature_cols) == 28, (
        f"Expected 28 feature columns, got {len(feature_cols)}: {feature_cols}"
    )

    X = df[feature_cols].copy()

    nanCounts = X.isnull().sum()
    if nanCounts.any():
        print(f"\nNaN counts per column:\n{nanCounts[nanCounts > 0].to_string()}")
    else:
        print("\nNo NaNs found initially.")

    # infMask = X.isin([np.inf, -np.inf])
    infMask = np.isinf(X.values)  # check for infinties before replacing
    nInf = infMask.sum()
    print(f"Infinity values found: {nInf}")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    rowsBefore = len(X)
    valid = X.notna().all(axis=1)
    # X = X.fillna(X.median())
    # X = X.fillna(0)
    X = X[valid]
    y = y[valid]
    rowsDropped = rowsBefore - len(X)
    print(f"Rows dropped (NaN after inf-replace): {rowsDropped}")


    print(f"\nFinal dataset shape: {X.shape}")
    print("Final class balance:")
    print(y.value_counts().rename({0: "0 (benign)", 1: "1 (malicious)"}).to_string())

    return X, y, feature_cols, benign_label


def preprocess(X, y, feature_cols, test_size, seed, output):
    print(f"\n{'='*60}")
    print("Preprocessing and splitting")
    print(f"{'='*60}")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(  # stratify keeps class ratio in both splits
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"Test  class dist: {dict(y_test.value_counts().sort_index())}")

    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    scaler = StandardScaler()  # fit on train only, dont leak test info
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(output, "scaler.joblib"))
    print(f"Scaler saved to {output}/scaler.joblib")


    malMask = (y_train == 1).values
    X_train_arr = X_train_sc
    X_mal = X_train_arr[malMask]
    # p10 = np.percentile(X_mal, 5, axis=0)
    # p90 = np.percentile(X_mal, 95, axis=0)
    p10 = np.percentile(X_mal, 10, axis=0)
    p90 = np.percentile(X_mal, 90, axis=0)
    feature_limits = np.stack([p10, p90], axis=1)
    np.save(os.path.join(output, "feature_limits.npy"), feature_limits)
    print(f"Feature limits saved: {feature_limits.shape}")

    corr = np.corrcoef(X_train_sc.T)
    np.save(os.path.join(output, "pearson_matrix.npy"), corr)
    with open(os.path.join(output, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"Pearson matrix saved: {corr.shape}")

    return X_train_sc, X_test_sc, y_train.values, y_test.values


class DoHMLP(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        # tried [256, 128, 64] -- overfit badly on small benign class
        # tried [64, 32] -- underfit, missed a lot of benign
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Dropout(0.5),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, seed, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  MLP device: {device}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)


    model = DoHMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # batch_size = 128
    # batch_size = 512
    batch_size = 256
    nBatches = int(np.ceil(len(X_tr) / batch_size))
    # patience = 5
    # patience = 15
    patience = 10
    bestValLoss = float("inf")
    noImprove = 0
    bestState = None

    for epoch in range(1, 101):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        epochLoss = 0.0
        for i in range(nBatches):
            batchIdx = idx[i * batch_size:(i + 1) * batch_size]
            xb, yb = X_tr_t[batchIdx], y_tr_t[batchIdx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        # if val_loss < bestValLoss - 1e-4:
        if val_loss < bestValLoss:  # save best chekpoint
            bestValLoss = val_loss
            noImprove = 0
            bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            noImprove += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  train_loss={epochLoss/nBatches:.4f}  "
                  f"val_loss={val_loss:.4f}  no_improve={noImprove}")

        if noImprove >= patience:  # stopp traning if no improvment
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(bestState)
    model.eval()
    return model, device


def train_all_models(X_train, y_train, seed, output, no_nn):
    print(f"\n{'='*60}")
    print("Model training")
    print(f"{'='*60}")

    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    scale_pos_weight = n0 / n1
    print(f"  class 0 (benign): {n0}, class 1 (malicious): {n1}")
    print(f"  scale_pos_weight = {scale_pos_weight:.4f}")

    models = {}

    print("\n[1/4] Random Forest")
    t0 = time.time()
    # rf = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=seed)
    # rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=seed
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    joblib.dump(rf, os.path.join(output, "rf.joblib"))
    models["rf"] = rf


    print("\n[2/4] Gradient Boosting")
    t0 = time.time()
    # gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed)
    # gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.7, random_state=seed)
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=seed
    )
    gb.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    joblib.dump(gb, os.path.join(output, "gb.joblib"))
    models["gb"] = gb


    print("\n[3/4] XGBoost")
    t0 = time.time()
    # xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.5, eval_metric="logloss", scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed)
    # xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.7, eval_metric="logloss", scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed)
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1, random_state=seed,
    )
    xgb.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    joblib.dump(xgb, os.path.join(output, "xgb.joblib"))
    models["xgb"] = xgb


    if not no_nn:
        print("\n[4/4] MLP (PyTorch)")
        t0 = time.time()
        mlp, device = train_mlp(X_train, y_train, seed, output)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")
        torch.save(mlp.state_dict(), os.path.join(output, "mlp.pt"))
        models["mlp"] = (mlp, device)
    else:
        print("\n[4/4] MLP skipped (--no_nn)")

    return models


def evaluate_model(name, y_true, y_pred, y_prob, output_dir):
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_bin = f1_score(y_true, y_pred, average="binary")
    roc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, target_names=["benign", "malicious"])

    header = f"\n{'='*60}\nModel: {name}\n{'='*60}"
    lines = [
        header,
        f"Accuracy:          {acc:.4f}",
        f"F1 (macro):        {f1_mac:.4f}",
        f"F1 (binary):       {f1_bin:.4f}",
        f"ROC-AUC:           {roc:.4f}",
        "Classification Report:",
        report,
    ]
    text = "\n".join(lines)
    print(text)

    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "a") as f:
        f.write(text + "\n")


    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["benign", "malicious"],
                yticklabels=["benign", "malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"), dpi=150)
    plt.close()


    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"), dpi=150)
    plt.close()


def run_evaluation(models, X_test, y_test, output, no_nn):
    print(f"\n{'='*60}")
    print("Evaluation")
    print(f"{'='*60}")

    open(os.path.join(output, "results.txt"), "w").close()

    for name in ("rf", "gb", "xgb"):
        model = models[name]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # grab the positive class probabilties
        evaluate_model(name.upper(), y_test, y_pred, y_prob, output)

    if not no_nn and "mlp" in models:
        mlp, device = models["mlp"]
        X_t = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            probs = mlp(X_t).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)  # threshhold at 0.5
        evaluate_model("MLP", y_test, preds, probs, output)


def run_shap(models, X_train, X_test, y_test, feature_cols, output, no_nn, seed):
    print(f"\n{'='*60}")
    print("SHAP analysis")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    # n_shap = min(200, len(X_test))
    # n_shap = min(1000, len(X_test))
    n_shap = min(500, len(X_test))
    shap_idx = rng.choice(len(X_test), size=n_shap, replace=False)
    X_shap = X_test[shap_idx]

    def save_shap_results(name, shap_vals, feat_names, output_dir):
        meanAbs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(meanAbs)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(feat_names)), meanAbs[order], align="center")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=7)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{name} — SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_shap_importance.png"), dpi=150)
        plt.close()

        shap_dict = {feat_names[i]: float(meanAbs[i]) for i in range(len(feat_names))}
        with open(os.path.join(output_dir, f"{name}_shap_values.json"), "w") as f:
            json.dump(shap_dict, f, indent=2)

    for key, display in (("rf", "RF"), ("gb", "GB"), ("xgb", "XGB")):
        print(f"\nSHAP for {display}...")
        try:
            explainer = shap.TreeExplainer(models[key])
            sv = explainer.shap_values(X_shap, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            save_shap_results(display, sv, feature_cols, output)
        except Exception as e:
            print(f"  SHAP {display} failed: {e}")

    if not no_nn and "mlp" in models:
        print("\nSHAP for MLP...")
        try:
            mlp, device = models["mlp"]
            # bg_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
            # bg_idx = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
            bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
            background = torch.FloatTensor(X_train[bg_idx]).to(device)  # basline for deep explainer
            X_shap_t = torch.FloatTensor(X_shap).to(device)
            # explainer = shap.GradientExplainer(mlp, background)
            explainer = shap.DeepExplainer(mlp, background)
            sv = explainer.shap_values(X_shap_t)
            if isinstance(sv, list):
                sv = sv[0]
            if isinstance(sv, torch.Tensor):
                sv = sv.cpu().numpy()
            save_shap_results("MLP", sv, feature_cols, output)
        except Exception as e:
            print(f"  SHAP MLP failed: {e}")


def per_tool_breakdown(l3_path, models, scaler, feature_cols, output, no_nn, sample, seed):
    print(f"\n{'='*60}")
    print("Per-tool breakdown (l3)")
    print(f"{'='*60}")

    df = pd.read_csv(l3_path, encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿") for c in df.columns]

    if sample is not None:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)

    label_col = find_label_col(df)
    tools = df[label_col].unique().tolist()
    print(f"Tools found: {tools}")

    X_l3 = df[feature_cols].copy()
    X_l3.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid = X_l3.notna().all(axis=1)
    df = df[valid]
    X_l3 = X_l3[valid]

    X_l3_sc = scaler.transform(X_l3)  # reuse fitted scaler, dont refit on l3
    labels_l3 = df[label_col].values

    results = {}
    modelKeys = ["rf", "gb", "xgb"]
    if not no_nn and "mlp" in models:
        modelKeys.append("mlp")

    for tool in sorted(tools):
        mask = labels_l3 == tool
        if mask.sum() == 0:
            continue
        X_tool = X_l3_sc[mask]
        y_tool = np.ones(mask.sum(), dtype=int)  # all malicous in l3


        row = {}
        for key in ("rf", "gb", "xgb"):
            if key not in models:
                continue
            y_pred = models[key].predict(X_tool)
            row[key.upper() + "_F1"] = f1_score(y_tool, y_pred, average="binary",
                                                  pos_label=1, zero_division=0)
            row[key.upper() + "_ACC"] = accuracy_score(y_tool, y_pred)

        if not no_nn and "mlp" in models:
            mlp, device = models["mlp"]
            X_t = torch.FloatTensor(X_tool).to(device)
            with torch.no_grad():
                probs = mlp(X_t).cpu().numpy().flatten()
            y_pred = (probs >= 0.5).astype(int)
            row["MLP_F1"] = f1_score(y_tool, y_pred, average="binary",
                                     pos_label=1, zero_division=0)
            row["MLP_ACC"] = accuracy_score(y_tool, y_pred)

        results[tool] = row
        print(f"  {tool}: {row}")


    colOrder = ["RF_F1", "GB_F1", "XGB_F1"]
    if not no_nn and "mlp" in models:
        colOrder.append("MLP_F1")

    header = f"{'Tool':<16}" + "".join(f"| {c:<8}" for c in colOrder)
    sep = "-" * len(header)
    lines = [header, sep]
    for tool, row in sorted(results.items()):
        line = f"{tool:<16}" + "".join(
            f"| {row.get(c, float('nan')):<8.4f}" for c in colOrder
        )
        lines.append(line)

    table = "\n".join(lines)
    print(f"\n{table}")

    outPath = os.path.join(output, "per_tool_results.txt")
    with open(outPath, "w") as f:
        f.write(table + "\n")
    print(f"\nPer-tool results saved to {outPath}")


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y, feature_cols, benign_label = load_and_inspect(
        args.l2, sample=args.sample, seed=args.seed
    )

    X_train, X_test, y_train, y_test = preprocess(
        X, y, feature_cols, args.test_size, args.seed, args.output
    )

    models = train_all_models(X_train, y_train, args.seed, args.output, args.no_nn)

    run_evaluation(models, X_test, y_test, args.output, args.no_nn)

    run_shap(models, X_train, X_test, y_test, feature_cols,
             args.output, args.no_nn, args.seed)

    if args.l3:
        scaler = joblib.load(os.path.join(args.output, "scaler.joblib"))
        per_tool_breakdown(
            args.l3, models, scaler, feature_cols,
            args.output, args.no_nn, args.sample, args.seed
        )

    print(f"\nAll models saved to {args.output}. "
          f"Results written to {args.output}/results.txt")
