"""
Adversarial Feedback Loop
==========================
The core research loop:

  1. Load or train a classifier (defender's model)
  2. Generate adversarial DoH flows using the C2 client
  3. Extract features the same way the defender does
  4. Score flows against the classifier
  5. Identify which features are leaking
  6. Adapt attack parameters
  7. Repeat — measuring evasion rate at each round

This gives you a rigorous, quantitative measure of:
  - How well each evasion strategy works
  - Which features are hardest to mimic
  - Where flow-based detection breaks down

Run this file to generate a full adversarial evaluation report.
"""

import numpy as np
import pandas as pd
import json
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from traffic_shaper import TrafficShaper, EvasionStrategy, QuerySchedule, FlowProfile
from doh_c2_client import DoHC2Client, C2Protocol


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FlowFeatureExtractor:
    """
    Extracts CIRA-CIC-compatible flow features from a QuerySchedule.
    This mirrors what a real network tap + CICFlowMeter would produce.

    Used to:
      (a) Feed generated flows into the classifier
      (b) Measure feature divergence from benign fingerprint
    """

    def extract(self, schedule: List[QuerySchedule]) -> dict:
        """Extract flow-level features from a generated query schedule."""
        delays = np.array([q.delay_ms for q in schedule])
        sizes  = np.array([q.payload_bytes + q.padding_bytes + 40 for q in schedule])
        cover_mask = np.array([q.is_cover for q in schedule])

        real_sizes = sizes[~cover_mask]
        real_delays = delays[~cover_mask]

        return {
            # Flow features
            "Duration":             float(delays.sum()),
            "FlowBytesSent":        float(sizes.sum()),
            "FlowSentRate":         float(sizes.sum() / max(delays.sum(), 1) * 1000),

            # Packet length features
            "PacketLengthMean":     float(sizes.mean()),
            "PacketLengthVariance": float(sizes.var()),
            "PacketLengthStandardDeviation": float(sizes.std()),
            "PacketLengthMedian":   float(np.median(sizes)),
            "PacketLengthSkewFromMedian": float(
                (sizes.mean() - np.median(sizes)) / (sizes.std() + 1e-9)
            ),
            "PacketLengthCoefficientofVariation": float(
                sizes.std() / (sizes.mean() + 1e-9)
            ),

            # Response time (IAT) features
            "ResponseTimeTimeMean":     float(delays.mean()),
            "ResponseTimeTimeVariance": float(delays.var()),
            "ResponseTimeTimeStandardDeviation": float(delays.std()),
            "ResponseTimeTimeMedian":   float(np.median(delays)),
            "ResponseTimeTimeSkewFromMedian": float(
                (delays.mean() - np.median(delays)) / (delays.std() + 1e-9)
            ),
            "ResponseTimeTimeCoefficientofVariation": float(
                delays.std() / (delays.mean() + 1e-9)
            ),

            # DNS-specific
            "DoHDNSQueryLength":   float(real_sizes.mean() if len(real_sizes) else 0),
            "DoHDNSAnswerCount":   float(np.random.uniform(1, 5)),  # sim
            "CoverQueryFraction":  float(cover_mask.mean()),
            "TotalQueryCount":     float(len(schedule)),
        }


# ── Stub Classifier (replace with your team's real model) ─────────────────────

class ClassifierInterface:
    """
    Interface to the defender's classifier.

    Replace _predict_proba() with your actual model.
    This stub uses a simple rule-based detector as placeholder.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_importances = {}

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print("[*] No model provided — using stub rule-based detector")
            print("[*] Replace ClassifierInterface._predict_proba() with your team's model")

    def _load_model(self, path: str):
        """Load a sklearn-compatible model (joblib or pickle)."""
        try:
            import joblib
            self.model = joblib.load(path)
            if hasattr(self.model, "feature_importances_"):
                print(f"[+] Loaded model from {path}")
        except Exception as e:
            print(f"[!] Failed to load model: {e}")

    def predict(self, features: dict) -> Tuple[str, float]:
        """
        Returns (label, confidence).
        label: "malicious" or "benign"
        confidence: 0.0 – 1.0
        """
        proba = self._predict_proba(features)
        label = "malicious" if proba > 0.5 else "benign"
        return label, proba

    def _predict_proba(self, features: dict) -> float:
        """
        Probability of being malicious. REPLACE THIS with your actual model.

        Current stub: simple threshold rules on the most discriminating features.
        This approximates an un-evaded baseline detector.
        """
        if self.model is not None:
            # Real model path
            import pandas as pd
            df = pd.DataFrame([features])
            return float(self.model.predict_proba(df)[:, 1][0])

        # ── Stub rules (approximate CIRA-CIC paper's findings) ────────────────
        score = 0.0

        # Rule 1: Fixed beacon interval (low IAT variance)
        iat_cv = features.get("ResponseTimeTimeCoefficientofVariation", 1.0)
        if iat_cv < 0.15:  # Very regular timing
            score += 0.45

        # Rule 2: Uniform packet sizes
        pkt_cv = features.get("PacketLengthCoefficientofVariation", 1.0)
        if pkt_cv < 0.10:
            score += 0.30

        # Rule 3: Unusually high query rate
        rate = features.get("FlowSentRate", 0)
        if rate > 5000:  # bytes/sec
            score += 0.20

        # Rule 4: Very short flow duration with high query count
        duration = features.get("Duration", 1)
        queries  = features.get("TotalQueryCount", 0)
        if queries > 20 and duration < 5000:
            score += 0.25

        return min(1.0, score)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Return feature importance scores (from the underlying model).
        Used by the adaptive attacker to prioritize what to fix.
        """
        if self.model and hasattr(self.model, "feature_importances_"):
            return dict(zip(
                self.model.feature_names_in_,
                self.model.feature_importances_
            ))
        # Stub importances based on CIRA-CIC paper
        return {
            "ResponseTimeTimeCoefficientofVariation": 0.28,
            "PacketLengthCoefficientofVariation":     0.22,
            "PacketLengthVariance":                   0.15,
            "ResponseTimeTimeVariance":               0.14,
            "FlowSentRate":                           0.09,
            "Duration":                               0.07,
            "DoHDNSQueryLength":                      0.05,
        }


# ── Adversarial Evaluator ──────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round:          int
    strategy:       str
    n_flows:        int
    detected:       int
    evasion_rate:   float
    mean_confidence:float
    top_leaking_feature: str
    leaking_ks_stat: float


class AdversarialEvaluator:
    """
    Orchestrates the attack-detect-adapt loop.

    Each round:
      1. Generate N adversarial flows
      2. Extract features
      3. Score against classifier
      4. Measure evasion rate
      5. Find leaking features
      6. Adapt shaper
    """

    def __init__(
        self,
        classifier:      ClassifierInterface,
        fingerprint_path: Optional[str] = None,
        n_flows_per_round: int = 100,
        payload_size:    int = 512,
    ):
        self.clf        = classifier
        self.shaper     = TrafficShaper(fingerprint_path)
        self.extractor  = FlowFeatureExtractor()
        self.n_flows    = n_flows_per_round
        self.payload    = b"RESEARCH:" + b"X" * (payload_size - 9)
        self.results:   List[RoundResult] = []
        self._benign_fp = self.shaper.fp

    def run(self, rounds: int = 6) -> pd.DataFrame:
        """
        Run the full adversarial loop for `rounds` iterations.
        Starts with NAIVE and escalates strategy, then does adaptive rounds.
        """
        strategies = [
            EvasionStrategy.NAIVE,
            EvasionStrategy.TIMING_ONLY,
            EvasionStrategy.SIZE_MIMICRY,
            EvasionStrategy.COVER_TRAFFIC,
            EvasionStrategy.FULL_MIMICRY,
            EvasionStrategy.ADAPTIVE,
        ]

        print("\n" + "=" * 65)
        print("  ADVERSARIAL EVALUATION — DoH C2 vs Flow-Based Classifier")
        print("=" * 65)

        for rnd in range(min(rounds, len(strategies))):
            strategy = strategies[rnd]
            print(f"\n── Round {rnd+1}: {strategy.value} ──────────────────────────")

            result = self._evaluate_strategy(rnd + 1, strategy)
            self.results.append(result)

            print(f"   Evasion rate:     {result.evasion_rate:.1%}")
            print(f"   Mean confidence:  {result.mean_confidence:.3f}")
            print(f"   Top leak:         {result.top_leaking_feature} (KS={result.leaking_ks_stat:.3f})")

            # Feed back into adaptive shaper
            if strategy == EvasionStrategy.ADAPTIVE:
                self.shaper.update_strategy({
                    "detected_rate":       1 - result.evasion_rate,
                    "top_leaking_feature": result.top_leaking_feature,
                    "leaking_feature_delta": result.leaking_ks_stat,
                })

        return self._results_dataframe()

    def _evaluate_strategy(
        self, rnd: int, strategy: EvasionStrategy
    ) -> RoundResult:
        """Evaluate a single strategy across N flows."""
        detected = 0
        confidences = []
        feature_records = []

        for _ in range(self.n_flows):
            # Generate adversarial flow
            schedule = self.shaper.build_schedule(
                self.payload, strategy=strategy, chunk_size=40
            )
            # Extract features
            features = self.extractor.extract(schedule)
            feature_records.append(features)

            # Classify
            label, conf = self.clf.predict(features)
            confidences.append(conf)
            if label == "malicious":
                detected += 1

        evasion_rate = 1 - detected / self.n_flows

        # Find the most leaking feature (biggest divergence from benign)
        top_feat, top_ks = self._find_leaking_feature(feature_records)

        return RoundResult(
            round=rnd,
            strategy=strategy.value,
            n_flows=self.n_flows,
            detected=detected,
            evasion_rate=evasion_rate,
            mean_confidence=float(np.mean(confidences)),
            top_leaking_feature=top_feat,
            leaking_ks_stat=top_ks,
        )

    def _find_leaking_feature(
        self, feature_records: List[dict]
    ) -> Tuple[str, float]:
        """Find which feature diverges most from benign fingerprint."""
        from scipy import stats

        df_attack = pd.DataFrame(feature_records)
        top_feat, top_ks = "unknown", 0.0

        for feat, benign_dist in self._benign_fp.items():
            if feat not in df_attack.columns:
                continue
            attack_vals = df_attack[feat].dropna().values
            if len(attack_vals) < 5:
                continue

            # Generate benign samples to compare against
            benign_samples = np.random.normal(
                benign_dist["mean"], benign_dist["std"], len(attack_vals)
            )
            benign_samples = np.clip(
                benign_samples,
                benign_dist.get("p5", 0),
                benign_dist.get("p95", 1e9)
            )

            ks, _ = stats.ks_2samp(benign_samples, attack_vals)
            if ks > top_ks:
                top_ks, top_feat = ks, feat

        return top_feat, top_ks

    def _results_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self.results])
        return df

    def print_summary(self):
        """Print a summary table of all rounds."""
        df = self._results_dataframe()
        print("\n" + "=" * 65)
        print("  RESULTS SUMMARY")
        print("=" * 65)
        print(df[["round", "strategy", "evasion_rate",
                   "mean_confidence", "top_leaking_feature"]].to_string(index=False))

    def save_report(self, path: str = "adversarial_report.json"):
        """Save full results to JSON."""
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rounds": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Report saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DoH Adversarial Evaluation Loop")
    parser.add_argument("--model",       default=None,      help="Path to classifier model (joblib)")
    parser.add_argument("--fingerprint", default=None,      help="Path to benign_fingerprint.json")
    parser.add_argument("--rounds",      type=int, default=6, help="Number of evaluation rounds")
    parser.add_argument("--flows",       type=int, default=50, help="Flows per round")
    parser.add_argument("--report",      default="adversarial_report.json")
    args = parser.parse_args()

    clf       = ClassifierInterface(model_path=args.model)
    evaluator = AdversarialEvaluator(
        classifier=clf,
        fingerprint_path=args.fingerprint,
        n_flows_per_round=args.flows,
    )

    results = evaluator.run(rounds=args.rounds)
    evaluator.print_summary()
    evaluator.save_report(args.report)

    print("\n[*] Next steps:")
    print("    1. Share adversarial_report.json with your team")
    print("    2. Team retrains classifier on adversarial flows")
    print("    3. Re-run this script — evasion rate should drop")
    print("    4. Adapt further → repeat")


if __name__ == "__main__":
    main()
