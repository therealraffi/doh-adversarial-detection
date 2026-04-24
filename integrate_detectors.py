"""
integrate_detectors.py
======================
Connects our adversarial framework directly to the team's trained models
from the 'detectors' branch (rf.joblib, gb.joblib, xgb.joblib, mlp.pt).

Usage:
    # First train the detectors (from detectors branch):
    python detector.py --l2 l2-total-add.csv --output ./results

    # Then run this from the adversarial branch:
    python integrate_detectors.py --results ./results

    # Or point directly at joblib files:
    python integrate_detectors.py --results ./results --flows 200
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_shaper import TrafficShaper, EvasionStrategy, QuerySchedule


# ── Feature names the detector uses (28 features from CIRA-CIC) ──────────────
# Loaded dynamically from results/feature_names.json if available.
# These are the exact column names detector.py trains on.

DEFAULT_28_FEATURES = [
    "Duration", "FlowBytesSent", "FlowSentRate", "FlowBytesReceived",
    "FlowReceivedRate", "PacketLengthVariance", "PacketLengthStandardDeviation",
    "PacketLengthMean", "PacketLengthMedian", "PacketLengthMode",
    "PacketLengthSkewFromMedian", "PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation", "DoHDNSQueryID", "DoHDNSQueryType",
    "DoHDNSQueryLength", "DoHDNSAnswerCount", "ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode", "ResponseTimeTimeCoefficientofVariation",
    "ResponseTimeTimeVariance", "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean", "ResponseTimeTimeMedian", "ResponseTimeTimeMode",
    "FlowSentRate", "PacketTimeMean", "PacketTimeStandardDeviation",
]


# ── Feature extractor (maps our schedule → 28 detector features) ─────────────

class FeatureExtractor:
    """
    Converts a QuerySchedule into the exact 28-feature vector
    the detector's models were trained on.
    """

    def extract(self, schedule: List[QuerySchedule], feature_names: List[str]) -> np.ndarray:
        delays = np.array([q.delay_ms for q in schedule])
        sizes  = np.array([q.payload_bytes + q.padding_bytes + 40 for q in schedule])
        real   = sizes[[not q.is_cover for q in schedule]]

        # Build full feature dict
        feat = {
            "Duration":                           float(delays.sum()),
            "FlowBytesSent":                      float(sizes.sum()),
            "FlowSentRate":                       float(sizes.sum() / max(delays.sum(), 1) * 1000),
            "FlowBytesReceived":                  float(sizes.sum() * np.random.uniform(1.1, 3.5)),
            "FlowReceivedRate":                   float(sizes.sum() * 1.8 / max(delays.sum(), 1) * 1000),
            "PacketLengthVariance":               float(sizes.var()),
            "PacketLengthStandardDeviation":      float(sizes.std()),
            "PacketLengthMean":                   float(sizes.mean()),
            "PacketLengthMedian":                 float(np.median(sizes)),
            "PacketLengthMode":                   float(sizes.mean()),  # approx
            "PacketLengthSkewFromMedian":         float((sizes.mean() - np.median(sizes)) / (sizes.std() + 1e-9)),
            "PacketLengthSkewFromMode":           float((sizes.mean() - sizes.mean()) / (sizes.std() + 1e-9)),
            "PacketLengthCoefficientofVariation": float(sizes.std() / (sizes.mean() + 1e-9)),
            "DoHDNSQueryID":                      float(np.random.randint(0, 65535)),
            "DoHDNSQueryType":                    float(np.random.choice([1, 28])),
            "DoHDNSQueryLength":                  float(real.mean() if len(real) else sizes.mean()),
            "DoHDNSAnswerCount":                  float(np.random.uniform(1, 5)),
            "ResponseTimeTimeSkewFromMedian":     float((delays.mean() - np.median(delays)) / (delays.std() + 1e-9)),
            "ResponseTimeTimeSkewFromMode":       float((delays.mean() - delays.mean()) / (delays.std() + 1e-9)),
            "ResponseTimeTimeCoefficientofVariation": float(delays.std() / (delays.mean() + 1e-9)),
            "ResponseTimeTimeVariance":           float(delays.var()),
            "ResponseTimeTimeStandardDeviation":  float(delays.std()),
            "ResponseTimeTimeMean":               float(delays.mean()),
            "ResponseTimeTimeMedian":             float(np.median(delays)),
            "ResponseTimeTimeMode":               float(delays.mean()),  # approx
            "PacketTimeMean":                     float(delays.mean()),
            "PacketTimeStandardDeviation":        float(delays.std()),
        }

        # Return only the features the model was trained on, in correct order
        vec = np.array([feat.get(f, 0.0) for f in feature_names], dtype=np.float32)
        return vec


# ── Model loader ──────────────────────────────────────────────────────────────

class DetectorLoader:
    """Loads trained models from the detectors branch results directory."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self._load()

    def _load(self):
        print(f"\n[*] Loading models from: {self.results_dir}")

        # Load feature names
        fn_path = os.path.join(self.results_dir, "feature_names.json")
        if os.path.exists(fn_path):
            with open(fn_path) as f:
                self.feature_names = json.load(f)
            print(f"[+] Feature names loaded: {len(self.feature_names)} features")
        else:
            self.feature_names = DEFAULT_28_FEATURES
            print(f"[!] feature_names.json not found. Using defaults ({len(self.feature_names)} features)")

        # Load scaler
        scaler_path = os.path.join(self.results_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[+] Scaler loaded from {scaler_path}")
        else:
            print(f"[!] No scaler found at {scaler_path} — will skip scaling")

        # Load sklearn models
        for name, fname in [("RF", "rf.joblib"), ("GB", "gb.joblib"), ("XGB", "xgb.joblib")]:
            path = os.path.join(self.results_dir, fname)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"[+] {name} loaded from {path}")
            else:
                print(f"[!] {name} not found at {path}")

        # Load MLP
        mlp_path = os.path.join(self.results_dir, "mlp.pt")
        if os.path.exists(mlp_path):
            try:
                import torch
                import torch.nn as nn

                class DoHMLP(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(32, 1), nn.Sigmoid(),
                        )
                    def forward(self, x): return self.net(x)

                mlp = DoHMLP(len(self.feature_names))
                mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
                mlp.eval()
                self.models["MLP"] = mlp
                print(f"[+] MLP loaded from {mlp_path}")
            except Exception as e:
                print(f"[!] MLP load failed: {e}")

        if not self.models:
            print("[!] No models loaded. Run detector.py first to train and save models.")

    def predict_proba(self, feature_vec: np.ndarray, model_name: str) -> float:
        """Returns P(malicious) for a feature vector."""
        model = self.models.get(model_name)
        if model is None:
            return 0.5

        X = feature_vec.reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)

        if model_name == "MLP":
            import torch
            with torch.no_grad():
                return float(model(torch.FloatTensor(X)).item())
        else:
            return float(model.predict_proba(X)[0, 1])


# ── Adversarial evaluator ─────────────────────────────────────────────────────

@dataclass
class Result:
    strategy:        str
    model:           str
    n_flows:         int
    detected:        int
    evasion_rate:    float
    mean_confidence: float


class AdversarialIntegration:
    """
    Runs all 6 evasion strategies against all loaded detector models
    and produces a results table.
    """

    def __init__(self, detector: DetectorLoader, fingerprint_path: Optional[str] = None):
        self.detector  = detector
        self.shaper    = TrafficShaper(fingerprint_path)
        self.extractor = FeatureExtractor()
        self.results: List[Result] = []

    def run(self, n_flows: int = 100, payload_size: int = 512) -> pd.DataFrame:
        payload = b"RESEARCH:" + b"X" * (payload_size - 9)

        strategies = list(EvasionStrategy)
        models     = list(self.detector.models.keys())

        if not models:
            print("[!] No models to evaluate against. Exiting.")
            return pd.DataFrame()

        print(f"\n{'='*65}")
        print(f"  ADVERSARIAL INTEGRATION — {len(strategies)} strategies × {len(models)} models")
        print(f"  {n_flows} flows per combination")
        print(f"{'='*65}")

        for strategy in strategies:
            print(f"\n── Strategy: {strategy.value} ──────────────────────")
            confidences_by_model: Dict[str, List[float]] = {m: [] for m in models}
            detected_by_model:    Dict[str, int]         = {m: 0  for m in models}

            for _ in range(n_flows):
                schedule = self.shaper.build_schedule(payload, strategy=strategy, chunk_size=40)
                feat_vec = self.extractor.extract(schedule, self.detector.feature_names)

                for model_name in models:
                    prob = self.detector.predict_proba(feat_vec, model_name)
                    confidences_by_model[model_name].append(prob)
                    if prob > 0.5:
                        detected_by_model[model_name] += 1

            for model_name in models:
                det   = detected_by_model[model_name]
                evasion = 1 - det / n_flows
                conf    = float(np.mean(confidences_by_model[model_name]))
                r = Result(
                    strategy=strategy.value,
                    model=model_name,
                    n_flows=n_flows,
                    detected=det,
                    evasion_rate=evasion,
                    mean_confidence=conf,
                )
                self.results.append(r)
                print(f"   {model_name:6s} → evasion={evasion:.1%}  conf={conf:.3f}")

        return self._dataframe()

    def _dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.results])

    def print_summary(self):
        df = self._dataframe()
        if df.empty:
            return
        print(f"\n{'='*65}")
        print("  FULL RESULTS — Evasion Rate per Strategy × Model")
        print(f"{'='*65}")

        # Pivot for readability
        pivot = df.pivot_table(
            index="strategy", columns="model", values="evasion_rate"
        )
        # Order strategies
        order = [s.value for s in EvasionStrategy]
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        print(pivot.applymap(lambda x: f"{x:.1%}").to_string())

    def save(self, path: str = "integration_results.json"):
        df = self._dataframe()
        df.to_json(path, orient="records", indent=2)
        df.to_csv(path.replace(".json", ".csv"), index=False)
        print(f"\n[+] Results saved to {path} and {path.replace('.json', '.csv')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Integrate adversarial framework with detector models")
    parser.add_argument("--results",     required=True, help="Path to detector results dir (contains *.joblib)")
    parser.add_argument("--fingerprint", default=None,  help="Path to benign_fingerprint.json")
    parser.add_argument("--flows",       type=int, default=100)
    parser.add_argument("--output",      default="integration_results.json")
    args = parser.parse_args()

    detector = DetectorLoader(args.results)
    runner   = AdversarialIntegration(detector, fingerprint_path=args.fingerprint)
    runner.run(n_flows=args.flows)
    runner.print_summary()
    runner.save(args.output)

    print("\n[*] Share integration_results.csv with your team for the presentation!")


if __name__ == "__main__":
    main()
