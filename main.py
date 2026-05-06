"""
DoH C2 Adversarial Research Framework
======================================
Network Security Research Project
FOR EDUCATIONAL / AUTHORIZED RESEARCH USE ONLY

QUICK START
-----------
1. Install dependencies:
   pip install numpy pandas scipy httpx scikit-learn joblib

2. (Optional) Download CIRA-CIC-DoHBrw-2020 dataset:
   https://www.unb.ca/cic/datasets/dohbrw-2020.html
   Then run:  python analysis/cira_cic_analyzer.py /path/to/dataset.csv
   This creates benign_fingerprint.json

3. Run the adversarial evaluation loop:
   python main.py

4. When your team has a trained classifier:
   python main.py --model /path/to/model.joblib

ARCHITECTURE
------------

  analysis/
    cira_cic_analyzer.py    ← Parses dataset, extracts benign distributions
                               Ranks features by detectability (KS statistic)

  evasion/
    traffic_shaper.py       ← Core evasion engine
                               Builds query schedules mimicking benign traffic
                               6 strategies: NAIVE → ADAPTIVE

  attacks/
    doh_c2_client.py        ← DoH C2 protocol implementation
                               Encodes, compresses, encrypts data into DNS
                               Sends via RFC 8484 DoH

  core/
    adversarial_loop.py     ← Attack–detect–adapt feedback loop
                               Measures evasion rate per strategy
                               Identifies leaking features
                               Feeds back to shaper

EVASION STRATEGIES (in order of sophistication)
------------------------------------------------
  NAIVE          → Fixed 5s beacon. No shaping. Trivially caught.
  TIMING_ONLY    → Randomized IAT matching benign distribution.
  SIZE_MIMICRY   → Timing + EDNS0 padding to match packet size distribution.
  COVER_TRAFFIC  → Size + interleaved decoy queries to real resolvers.
  FULL_MIMICRY   → Everything + burst grouping to simulate page loads.
  ADAPTIVE       → Full mimicry + classifier feedback to tighten parameters.

KEY RESEARCH QUESTIONS THIS ANSWERS
------------------------------------
  Q: How well do flow-based classifiers detect DoH C2?
  A: Run NAIVE → measure detection rate (baseline)

  Q: Which evasion strategies are most effective?
  A: Compare evasion rates across all strategies

  Q: Which features are hardest to mimic?
  A: top_leaking_feature column in adversarial_report.json

  Q: Can adaptive adversaries beat retrained classifiers?
  A: Run loop after team retrains on adversarial samples

CONNECTING TO YOUR TEAM'S CLASSIFIER
--------------------------------------
In core/adversarial_loop.py, ClassifierInterface._predict_proba():
  - Replace the stub with your actual sklearn model
  - Or load a joblib model via --model flag
  - The interface expects predict_proba() returning P(malicious)
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cira_cic_analyzer import get_synthetic_benign_fingerprint
from traffic_shaper import TrafficShaper, EvasionStrategy
from doh_c2_client import DoHC2Client
from adversarial_loop import AdversarialEvaluator, ClassifierInterface
import asyncio


def setup_fingerprint(dataset_path: str = None):
    """Create fingerprint from dataset or use synthetic."""
    fp_path = "benign_fingerprint.json"

    if dataset_path and os.path.exists(dataset_path):
        from cira_cic_analyzer import CIRACICAnalyzer
        analyzer = CIRACICAnalyzer(dataset_path)
        analyzer.print_top_detectable_features()
        analyzer.save_fingerprint(fp_path)
    elif not os.path.exists(fp_path):
        print("[*] No dataset provided. Generating synthetic fingerprint...")
        fp = get_synthetic_benign_fingerprint()
        with open(fp_path, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"[+] Synthetic fingerprint saved to {fp_path}")

    return fp_path


def run_quick_demo():
    """Show what the traffic shaper produces across all strategies."""
    print("\n── Quick Demo: Traffic Shaping ──────────────────────────────")
    shaper = TrafficShaper()
    payload = b"DEMO:" + b"X" * 200

    for s in EvasionStrategy:
        schedule = shaper.build_schedule(payload, strategy=s, chunk_size=40)
        profile  = shaper.summarize_schedule(schedule)
        iats     = [q.delay_ms for q in schedule]
        import numpy as np
        cv       = np.std(iats) / (np.mean(iats) + 1e-9)
        print(f"\n  {s.value:15s} | queries={profile.total_queries:3d} "
              f"| IAT_cv={cv:.2f} | size={profile.mean_query_size:.0f}B "
              f"| duration={profile.duration_ms/1000:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="DoH C2 Adversarial Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset",     default=None,  help="Path to CIRA-CIC CSV file")
    parser.add_argument("--model",       default=None,  help="Path to classifier model (.joblib)")
    parser.add_argument("--rounds",      type=int, default=6)
    parser.add_argument("--flows",       type=int, default=50,  help="Flows per round")
    parser.add_argument("--report",      default="adversarial_report.json")
    parser.add_argument("--demo",        action="store_true",   help="Run quick shaping demo only")
    parser.add_argument("--doh-demo",    action="store_true",   help="Run DoH C2 dry-run demo")
    args = parser.parse_args()

    print("=" * 65)
    print("  DoH C2 Adversarial Research Framework")
    print("  Network Security Research — Adversarial Role")
    print("=" * 65)

    if args.demo:
        run_quick_demo()
        return

    if args.doh_demo:
        from doh_c2_client import demo
        asyncio.run(demo())
        return

    # Full adversarial loop
    fp_path  = setup_fingerprint(args.dataset)
    clf      = ClassifierInterface(model_path=args.model)
    evaluator = AdversarialEvaluator(
        classifier=clf,
        fingerprint_path=fp_path,
        n_flows_per_round=args.flows,
    )

    results = evaluator.run(rounds=args.rounds)
    evaluator.print_summary()
    evaluator.save_report(args.report)


if __name__ == "__main__":
    main()
