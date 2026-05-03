"""
evaluate_real_c2.py
====================
Evaluates Raffi's real Sliver C2 PCAPs against our trained detectors.
This answers: do our models generalize to real C2 tools?
"""

import os, json, warnings
import numpy as np, pandas as pd, joblib

DETECTOR_FEATURES = [
    "FlowBytesSent","FlowSentRate","FlowBytesReceived","FlowReceivedRate",
    "PacketLengthVariance","PacketLengthStandardDeviation","PacketLengthMean",
    "PacketLengthMedian","PacketLengthMode","PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode","PacketLengthCoefficientofVariation",
    "PacketTimeVariance","PacketTimeStandardDeviation","PacketTimeMean",
    "PacketTimeMedian","PacketTimeMode","PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode","PacketTimeCoefficientofVariation",
    "ResponseTimeTimeVariance","ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean","ResponseTimeTimeMedian","ResponseTimeTimeMode",
    "ResponseTimeTimeSkewFromMedian","ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
]

PCAP_FILES = [
    ("01_quiet_beacon_flows.csv",  1, "C2 Quiet Beacon"),
    ("02_discovery_flows.csv",     1, "C2 Discovery"),
    ("03_exfiltration_flows.csv",  1, "C2 Exfiltration"),
    ("04_benign_web_flows.csv",    0, "Benign Web"),
    ("05_lateral_scan_flows.csv",  1, "C2 Lateral Scan"),
]

def map_features(df):
    rows = []
    for _, row in df.iterrows():
        pm  = row.get("pkt_len_mean", row.get("pkt_size_avg", 0))
        ps  = row.get("pkt_len_std", 0)
        pv  = ps**2
        px  = row.get("pkt_len_max", pm); pn = row.get("pkt_len_min", pm)
        im  = row.get("flow_iat_mean", 0) * 1000
        is_ = row.get("flow_iat_std", 0) * 1000
        iv  = is_**2
        ix  = row.get("flow_iat_max", 0) * 1000
        ino = row.get("flow_iat_min", 0) * 1000
        rm  = row.get("fwd_iat_mean", 0) * 1000
        rs  = row.get("fwd_iat_std", 0) * 1000
        rv  = rs**2
        rx  = row.get("fwd_iat_max", 0) * 1000
        rn  = row.get("fwd_iat_min", 0) * 1000
        fm  = row.get("fwd_pkt_len_min", pm)
        feat = {
            "FlowBytesSent":      row.get("totlen_fwd_pkts", 0),
            "FlowBytesReceived":  row.get("totlen_bwd_pkts", 0),
            "FlowSentRate":       row.get("flow_byts_s", 0),
            "FlowReceivedRate":   row.get("flow_pkts_s", 0),
            "PacketLengthMean":   pm, "PacketLengthVariance": pv,
            "PacketLengthStandardDeviation": ps,
            "PacketLengthMedian": (px+pn)/2, "PacketLengthMode": fm,
            "PacketLengthSkewFromMedian": (pm-(px+pn)/2)/(ps+1e-9),
            "PacketLengthSkewFromMode":   (pm-fm)/(ps+1e-9),
            "PacketLengthCoefficientofVariation": ps/(pm+1e-9),
            "PacketTimeMean":     im, "PacketTimeVariance": iv,
            "PacketTimeStandardDeviation": is_,
            "PacketTimeMedian":   (ix+ino)/2, "PacketTimeMode": im,
            "PacketTimeSkewFromMedian": (im-(ix+ino)/2)/(is_+1e-9),
            "PacketTimeSkewFromMode":   0.0,
            "PacketTimeCoefficientofVariation": is_/(im+1e-9),
            "ResponseTimeTimeMean":     rm, "ResponseTimeTimeVariance": rv,
            "ResponseTimeTimeStandardDeviation": rs,
            "ResponseTimeTimeMedian":   (rx+rn)/2, "ResponseTimeTimeMode": rm,
            "ResponseTimeTimeSkewFromMedian": (rm-(rx+rn)/2)/(rs+1e-9),
            "ResponseTimeTimeSkewFromMode":   0.0,
            "ResponseTimeTimeCoefficientofVariation": rs/(rm+1e-9),
        }
        rows.append([feat.get(f, 0.0) for f in DETECTOR_FEATURES])
    return np.array(rows, dtype=np.float32) if rows else None


def main():
    # Load models
    results_dir = "results_full"
    fn_path = os.path.join(results_dir, "feature_names.json")
    fn = json.load(open(fn_path)) if os.path.exists(fn_path) else DETECTOR_FEATURES

    scaler = joblib.load(os.path.join(results_dir, "scaler.joblib"))
    models = {}
    for name, fname in [("RF","rf.joblib"),("GB","gb.joblib"),("XGB","xgb.joblib")]:
        p = os.path.join(results_dir, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
            print(f"[+] {name} loaded")

    print(f"\n{'='*65}")
    print("  REAL C2 EVALUATION — Raffi's Sliver PCAPs")
    print(f"{'='*65}\n")

    results = []
    for csv_file, true_label, description in PCAP_FILES:
        if not os.path.exists(csv_file):
            print(f"[!] Missing: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

        X = map_features(df)
        if X is None or len(X) == 0:
            print(f"[!] No flows in {csv_file}")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = scaler.transform(X)

        print(f"{'─'*50}")
        print(f"  {description} (true label: {'Malicious' if true_label==1 else 'Benign'})")
        print(f"  Flows extracted: {len(X)}")

        for mn, model in models.items():
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]
            detected = int(preds.sum())
            correct = int((preds == true_label).sum())
            accuracy = correct / len(preds)

            status = "✓ CORRECT" if accuracy >= 0.5 else "✗ MISSED"
            print(f"  {mn:6s} → detected {detected}/{len(preds)} flows as malicious  "
                  f"conf={probs.mean():.3f}  {status}")

            results.append({
                "pcap": description,
                "true_label": true_label,
                "model": mn,
                "flows": len(preds),
                "detected_malicious": detected,
                "accuracy": accuracy,
                "mean_confidence": float(probs.mean()),
            })

    # Summary
    df_out = pd.DataFrame(results)
    df_out.to_csv("real_c2_evaluation.csv", index=False)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for desc, true_label in [(d, t) for _, t, d in PCAP_FILES]:
        subset = df_out[df_out["pcap"] == desc]
        if subset.empty: continue
        label_str = "Malicious" if true_label == 1 else "Benign"
        accs = subset.set_index("model")["accuracy"]
        print(f"  {desc:30s} ({label_str:8s}): "
              + "  ".join([f"{m}={accs.get(m,0):.0%}" for m in ["RF","GB","XGB"]]))

    print(f"\n[+] Saved to real_c2_evaluation.csv")


if __name__ == "__main__":
    main()
