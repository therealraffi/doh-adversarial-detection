"""
CIRA-CIC-DoHBrw-2020 Feature Analyzer
======================================
Loads the dataset and extracts statistical distributions of benign DoH traffic.
These distributions become the TARGET for our adversarial mimicry.

Dataset labels:
  - 'Benign-Firefox-Normal'  -> what we want to look like
  - 'Malicious-*'            -> what we are (and must hide)

Download: https://www.unb.ca/cic/datasets/dohbrw-2020.html
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats


# ── CIRA-CIC column names (from the dataset README) ──────────────────────────

FLOW_FEATURES = [
    "Duration",
    "FlowBytesSent",
    "FlowSentRate",
    "FlowBytesReceived",
    "FlowReceivedRate",
]

PACKET_FEATURES = [
    "PacketLengthVariance",
    "PacketLengthStandardDeviation",
    "PacketLengthMean",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation",
]

REQUEST_FEATURES = [
    "DoHDNSQueryID",
    "DoHDNSQueryType",
    "DoHDNSQueryLength",
    "DoHDNSAnswerCount",
]

TIMING_FEATURES = [
    "ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeMode",
]

ALL_FEATURES = FLOW_FEATURES + PACKET_FEATURES + REQUEST_FEATURES + TIMING_FEATURES


# ── Analyzer class ─────────────────────────────────────────────────────────────

class CIRACICAnalyzer:
    """
    Parse CIRA-CIC CSV files, separate benign vs malicious flows,
    and extract the statistical fingerprint of legitimate DoH traffic.
    """

    def __init__(self, csv_path: str):
        print(f"[*] Loading dataset from: {csv_path}")
        self.df = pd.read_csv(csv_path, low_memory=False)
        self._clean()
        self.benign = self.df[self.df["Label"].str.contains("Benign", case=False, na=False)]
        self.malicious = self.df[~self.df["Label"].str.contains("Benign", case=False, na=False)]
        print(f"[+] Benign flows:    {len(self.benign):,}")
        print(f"[+] Malicious flows: {len(self.malicious):,}")

    def _clean(self):
        # Drop NaN and Inf
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

    def get_benign_distributions(self) -> dict:
        """
        For each key feature, fit a distribution to benign traffic.
        Returns a dict of {feature: {mean, std, min, max, p5, p95, kde}}
        """
        distributions = {}
        available = [f for f in ALL_FEATURES if f in self.benign.columns]

        for feat in available:
            series = self.benign[feat].dropna()
            if len(series) < 10:
                continue
            distributions[feat] = {
                "mean":   float(series.mean()),
                "std":    float(series.std()),
                "min":    float(series.min()),
                "max":    float(series.max()),
                "p5":     float(np.percentile(series, 5)),
                "p25":    float(np.percentile(series, 25)),
                "p50":    float(np.percentile(series, 50)),
                "p75":    float(np.percentile(series, 75)),
                "p95":    float(np.percentile(series, 95)),
                "skew":   float(series.skew()),
            }

        print(f"[+] Extracted distributions for {len(distributions)} features")
        return distributions

    def compare_distributions(self, feature: str) -> dict:
        """
        KS-test: how different are benign vs malicious for a given feature?
        High KS statistic = easy to detect. Low = hard to detect.
        """
        if feature not in self.benign.columns:
            return {}

        b = self.benign[feature].dropna()
        m = self.malicious[feature].dropna()
        ks_stat, p_val = stats.ks_2samp(b, m)

        return {
            "feature":    feature,
            "ks_stat":    round(ks_stat, 4),   # 0=identical, 1=totally different
            "p_value":    round(p_val, 6),
            "detectable": ks_stat > 0.2,        # rough threshold
            "benign_mean":   round(float(b.mean()), 4),
            "malicious_mean": round(float(m.mean()), 4),
            "delta_pct":  round(abs(b.mean() - m.mean()) / (b.mean() + 1e-9) * 100, 2),
        }

    def rank_detection_features(self) -> pd.DataFrame:
        """
        Rank all features by how separable benign vs malicious are.
        These are the features the defender will use — and we must evade.
        """
        results = []
        available = [f for f in ALL_FEATURES if f in self.df.columns]

        for feat in available:
            r = self.compare_distributions(feat)
            if r:
                results.append(r)

        df = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
        return df

    def save_fingerprint(self, output_path: str = "benign_fingerprint.json"):
        """Save benign distributions to JSON for use by the attack engine."""
        dists = self.get_benign_distributions()
        with open(output_path, "w") as f:
            json.dump(dists, f, indent=2)
        print(f"[+] Saved fingerprint to: {output_path}")
        return dists

    def print_top_detectable_features(self, n: int = 15):
        """Print the features that most clearly separate benign from malicious."""
        df = self.rank_detection_features()
        print(f"\n{'='*65}")
        print(f"  TOP {n} MOST DETECTABLE FEATURES (highest KS statistic)")
        print(f"  These are what the classifier will likely key on")
        print(f"{'='*65}")
        print(df.head(n).to_string(index=False))
        print()
        return df


# ── Synthetic fingerprint (fallback if no dataset downloaded yet) ─────────────

def get_synthetic_benign_fingerprint() -> dict:
    """
    Approximate benign DoH fingerprint derived from published CIRA-CIC stats.
    Use this to bootstrap before you have the real dataset.
    Values based on: Montazerishatoori et al. (2020) + literature review.
    """
    return {
        "Duration": {
            "mean": 1842.3, "std": 4201.1, "p5": 12.0, "p95": 9800.0,
            "min": 1.0, "max": 86400000.0
        },
        "PacketLengthMean": {
            "mean": 387.4, "std": 201.3, "p5": 80.0, "p95": 850.0,
            "min": 40.0, "max": 1500.0
        },
        "PacketLengthVariance": {
            "mean": 52800.0, "std": 41200.0, "p5": 800.0, "p95": 145000.0,
            "min": 0.0, "max": 490000.0
        },
        "PacketLengthStandardDeviation": {
            "mean": 189.3, "std": 98.7, "p5": 28.0, "p95": 381.0,
            "min": 0.0, "max": 700.0
        },
        "DoHDNSQueryLength": {
            "mean": 32.4, "std": 18.7, "p5": 10.0, "p95": 78.0,
            "min": 5.0, "max": 253.0
        },
        "DoHDNSAnswerCount": {
            "mean": 2.8, "std": 2.1, "p5": 1.0, "p95": 8.0,
            "min": 0.0, "max": 30.0
        },
        "ResponseTimeTimeMean": {
            "mean": 148.2, "std": 312.4, "p5": 18.0, "p95": 620.0,
            "min": 1.0, "max": 5000.0
        },
        "ResponseTimeTimeVariance": {
            "mean": 28900.0, "std": 61200.0, "p5": 10.0, "p95": 95000.0,
            "min": 0.0, "max": 1200000.0
        },
        "FlowBytesSent": {
            "mean": 4820.0, "std": 8340.0, "p5": 120.0, "p95": 22000.0,
            "min": 40.0, "max": 500000.0
        },
        "FlowBytesReceived": {
            "mean": 6230.0, "std": 10800.0, "p5": 200.0, "p95": 28000.0,
            "min": 40.0, "max": 800000.0
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        analyzer = CIRACICAnalyzer(sys.argv[1])
        analyzer.print_top_detectable_features()
        analyzer.save_fingerprint("benign_fingerprint.json")
    else:
        print("[!] No dataset path provided. Generating synthetic fingerprint...")
        fp = get_synthetic_benign_fingerprint()
        with open("benign_fingerprint.json", "w") as f:
            json.dump(fp, f, indent=2)
        print("[+] Saved synthetic fingerprint to benign_fingerprint.json")
        print("[*] Run with: python cira_cic_analyzer.py /path/to/CIRA-CIC.csv")
