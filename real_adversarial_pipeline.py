"""
real_adversarial_pipeline.py  (v4 — Real Distribution Targeting)
==================================================================
Uses ACTUAL benign feature distributions from l2-total-add.csv:
  PacketLengthMode   mean=74B  (not 469B like we had before)
  PacketLengthMean   mean=137B
  PacketLengthMedian mean=94B
  PacketTimeStd      mean=11ms (not 440ms)
  FlowBytesSent      mean=10KB

This directly attacks the top SHAP features with correct values.
"""

import os, sys, time, random, subprocess, json, argparse, warnings, tempfile
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_shaper import TrafficShaper, EvasionStrategy
from benign_target_sampling import (
    CorrelatedBenignSampler,
    InterpolatedBenignSampler,
    run_tabular_correlation_validation,
    traffic_params_from_cira_features,
)

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("[!] pip install scapy")

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

# Columns compared to benign reference for optional realism gating (marginals).
REALISM_FEATURES = [
    "FlowBytesSent",
    "FlowBytesReceived",
    "FlowSentRate",
    "FlowReceivedRate",
    "PacketLengthMean",
    "PacketLengthStandardDeviation",
    "PacketLengthVariance",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthCoefficientofVariation",
    "PacketTimeMean",
    "PacketTimeStandardDeviation",
    "PacketTimeVariance",
    "PacketTimeMedian",
    "PacketTimeCoefficientofVariation",
    "ResponseTimeTimeMean",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeCoefficientofVariation",
]


def resolve_reference_data(path: str | None) -> str:
    if path and os.path.isfile(os.path.expanduser(path)):
        return os.path.abspath(os.path.expanduser(path))
    for rel in (
        "L2-BenignDoH-MaliciousDoH.parquet",
        os.path.join("data", "l2-total-add.csv"),
        "l2-total-add.csv",
    ):
        if os.path.isfile(rel):
            return os.path.abspath(rel)
    return os.path.join("data", "l2-total-add.csv")


def load_realism_profile(
    dataset_path: str, q_low: float = 0.01, q_high: float = 0.99
) -> dict:
    """Per-feature benign marginals (quantile band + mean/std) for gating."""
    p = resolve_reference_data(dataset_path)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".parquet":
        real = pd.read_parquet(p)
    else:
        real = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    real.columns = [c.lstrip("\ufeff") for c in real.columns]
    benign = real[real["Label"].astype(str).str.contains("Benign", case=False, na=False)]
    ref = {}
    q_low = float(np.clip(q_low, 0.0, 0.49))
    q_high = float(np.clip(q_high, 0.51, 1.0))
    if q_high <= q_low:
        q_low, q_high = 0.01, 0.99
    for c in REALISM_FEATURES:
        if c not in benign.columns:
            continue
        s = pd.to_numeric(benign[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 50:
            continue
        ref[c] = {
            "lo": float(s.quantile(q_low)),
            "hi": float(s.quantile(q_high)),
            "mean": float(s.mean()),
            "std": float(max(s.std(), 1e-9)),
        }
    return ref


def score_realism(feat: dict, ref: dict, z_max: float = 5.0) -> tuple[int, int, int]:
    """Returns (quantile_band_violation_count, z_violation_count, n_features_used)."""
    band_v = z_v = 0
    n = 0
    for c, st in ref.items():
        n += 1
        v = float(feat.get(c, 0.0))
        if v < st["lo"] or v > st["hi"]:
            band_v += 1
        z = abs(v - st["mean"]) / st["std"]
        if z > z_max:
            z_v += 1
    return band_v, z_v, n


def row_dict_from_matrix(m: np.ndarray, fn: list) -> dict:
    if len(m) == 0:
        return {}
    return {fn[j]: float(m[0, j]) for j in range(len(fn))}


# ── Real benign distributions (from actual dataset) ────────────────────────────
# PacketLengthMode   mean=74,  std=20,  p5=54,   p95=105
# PacketLengthMean   mean=137, std=82,  p5=73,   p95=254
# PacketLengthMedian mean=94,  std=17,  p5=66,   p95=123
# PacketTimeStd      mean=11,  std=18,  p5=0,    p95=58
# FlowBytesSent      mean=10KB,std=30KB,p5=128B, p95=68KB

def sample_packet_size():
    """
    Sample from real benign DoH packet size distribution.
    Most packets are small (DNS queries ~54-105B).
    Occasionally larger responses.
    """
    if random.random() < 0.6:
        # Small query packets (mode cluster ~74B)
        return max(40, int(random.gauss(74, 20)))
    else:
        # Larger response packets
        return max(100, int(random.gauss(220, 80)))

def sample_iat_ms():
    """
    Sample IAT matching benign PacketTimeStd=11ms.
    Very tight distribution — benign DoH is fast.
    """
    # Mean ~50ms, std ~11ms to match benign
    return max(5, random.gauss(50, 11))

def sample_n_packets():
    """
    Sample number of request/response pairs to match FlowBytesSent ~10KB.
    With ~74B per req + ~220B per resp = ~294B per pair.
    Need ~34 pairs for 10KB. Use 20-60 range.
    """
    return random.randint(20, 80)


# ── TCP session builders ──────────────────────────────────────────────────────

def tcp_handshake(sm, dm, si, di, sp, dp, t):
    p=[]
    syn=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='S',seq=1000)
    syn.time=t; p.append(syn); t+=random.uniform(0.001,0.005)
    sa=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/TCP(sport=dp,dport=sp,flags='SA',seq=2000,ack=1001)
    sa.time=t; p.append(sa); t+=random.uniform(0.001,0.005)
    ack=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='A',seq=1001,ack=2001)
    ack.time=t; p.append(ack); t+=random.uniform(0.001,0.005)
    return p, t, 1001, 2001

def tcp_teardown(sm, dm, si, di, sp, dp, t, sc, ss):
    p=[]
    f=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='FA',seq=sc,ack=ss)
    f.time=t; p.append(f); t+=random.uniform(0.001,0.005)
    fa=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/TCP(sport=dp,dport=sp,flags='FA',seq=ss,ack=sc+1)
    fa.time=t; p.append(fa); t+=random.uniform(0.001,0.005)
    ack=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='A',seq=sc+1,ack=ss+1)
    ack.time=t; p.append(ack)
    return p


def schedule_to_session(schedule, start_time, traffic_params=None):
    """
    Build TCP session. If traffic_params is set (from interpolated benign targets),
    IAT and sizes follow those stats; else legacy random benign-like sampling.
    """
    sm = "00:1A:2B:3C:4D:5E"
    si = "192.168.1.50"
    sp = random.randint(1024, 65535)
    dp = 443
    all_pkts = []

    def pad_iat_ms():
        if traffic_params:
            m = traffic_params["iat_ms_mean"]
            s = traffic_params["iat_ms_std"]
            return max(1.0, random.gauss(m, s))
        return sample_iat_ms()

    def req_wire():
        if traffic_params:
            return max(
                40,
                int(random.gauss(traffic_params["req_wire_mean"], traffic_params["req_wire_std"])),
            )
        return sample_packet_size()

    def resp_wire():
        if traffic_params:
            return max(
                60,
                int(random.gauss(traffic_params["resp_wire_mean"], traffic_params["resp_wire_std"])),
            )
        return max(60, int(random.gauss(200, 80)))

    real_q = [q for q in schedule if not q.is_cover]
    cover_q = [q for q in schedule if q.is_cover]

    # ── C2 session ────────────────────────────────────────────────────────────
    if real_q:
        di = "104.16.248.249"
        dm = "AA:BB:CC:DD:EE:11"
        t = start_time

        hs, t, sc, ss = tcp_handshake(sm, dm, si, di, sp, dp, t)
        all_pkts.extend(hs)

        if traffic_params:
            n_packets = max(len(real_q), int(traffic_params["n_pairs_c2"]))
        else:
            n_packets = max(len(real_q), random.randint(400, 600))

        for i in range(n_packets):
            if i < len(real_q):
                t += real_q[i].delay_ms / 1000.0
            else:
                t += pad_iat_ms() / 1000.0

            req_size = req_wire()
            req = (
                Ether(src=sm, dst=dm)
                / IP(src=si, dst=di)
                / TCP(sport=sp, dport=dp, flags="PA", seq=sc, ack=ss)
                / Raw(b"E" * max(1, req_size - 54))
            )
            req.time = t
            all_pkts.append(req)
            sc += req_size

            t += pad_iat_ms() / 1000.0
            resp_size = resp_wire()
            resp = (
                Ether(src=dm, dst=sm)
                / IP(src=di, dst=si)
                / TCP(sport=dp, dport=sp, flags="PA", seq=ss, ack=sc)
                / Raw(b"V" * max(1, resp_size - 54))
            )
            resp.time = t
            all_pkts.append(resp)
            ss += resp_size

        td = tcp_teardown(sm, dm, si, di, sp, dp, t, sc, ss)
        all_pkts.extend(td)

    # ── Cover sessions ────────────────────────────────────────────────────────
    for q in cover_q:
        cdi = "8.8.8.8"
        cdm = "AA:BB:CC:DD:EE:00"
        csp = random.randint(1024, 65535)
        ct = start_time + q.delay_ms / 1000.0
        hs, ct, sc2, ss2 = tcp_handshake(sm, cdm, si, cdi, csp, dp, ct)
        all_pkts.extend(hs)

        if traffic_params:
            n_cover = int(traffic_params["n_pairs_cover"])
        else:
            n_cover = random.randint(400, 600)
        for _ in range(n_cover):
            ct += pad_iat_ms() / 1000.0
            rs = req_wire()
            req = (
                Ether(src=sm, dst=cdm)
                / IP(src=si, dst=cdi)
                / TCP(sport=csp, dport=dp, flags="PA", seq=sc2, ack=ss2)
                / Raw(b"B" * max(1, rs - 54))
            )
            req.time = ct
            all_pkts.append(req)
            sc2 += rs
            ct += pad_iat_ms() / 1000.0
            rrs = resp_wire()
            resp = (
                Ether(src=cdm, dst=sm)
                / IP(src=cdi, dst=si)
                / TCP(sport=dp, dport=csp, flags="PA", seq=ss2, ack=sc2)
                / Raw(b"G" * max(1, rrs - 54))
            )
            resp.time = ct
            all_pkts.append(resp)
            ss2 += rrs

        td = tcp_teardown(sm, cdm, si, cdi, csp, dp, ct, sc2, ss2)
        all_pkts.extend(td)

    all_pkts.sort(key=lambda p: p.time)
    return all_pkts


# ── CICFlowMeter ─────────────────────────────────────────────────────────────

def _run_cicflowmeter_cli(pcap: str, csv_out: str) -> pd.DataFrame:
    try:
        subprocess.run(
            ["cicflowmeter", "-f", pcap, "-c", csv_out],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except Exception as e:
        print(f"[!] cicflowmeter CLI: {e}")
        return pd.DataFrame()
    if not os.path.exists(csv_out):
        return pd.DataFrame()
    df = pd.read_csv(csv_out)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def run_cicflowmeter(pcap: str, csv_out: str) -> pd.DataFrame:
    """Prefer in-process FlowSession (works on Windows); CLI if forced or on failure."""
    if os.environ.get("DOH_CFM_FORCE_CLI") == "1":
        return _run_cicflowmeter_cli(pcap, csv_out)
    try:
        from cicflowmeter.flow_session import FlowSession
        from scapy.all import rdpcap

        if os.path.exists(csv_out):
            os.remove(csv_out)
        fs = FlowSession(output_mode="csv", output=csv_out)
        for pkt in rdpcap(pcap):
            fs.process(pkt)
        fs.flush_flows()
    except Exception as e:
        if os.environ.get("DOH_CFM_DEBUG"):
            print(f"[!] CIC in-process failed ({e}); using CLI")
        return _run_cicflowmeter_cli(pcap, csv_out)
    if not os.path.exists(csv_out):
        return pd.DataFrame()
    df = pd.read_csv(csv_out)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def map_features(df, fn):
    rows=[]
    for _,row in df.iterrows():
        pm=row.get("pkt_len_mean",row.get("pkt_size_avg",0))
        ps=row.get("pkt_len_std",0)
        pv=row.get("pkt_len_var",ps**2)
        px=row.get("pkt_len_max",pm); pn=row.get("pkt_len_min",pm)
        im=row.get("flow_iat_mean",0)*1000; is_=row.get("flow_iat_std",0)*1000
        iv=is_**2
        ix=row.get("flow_iat_max",im/1000)*1000; ino=row.get("flow_iat_min",im/1000)*1000
        rm=row.get("fwd_iat_mean",im/1000)*1000; rs=row.get("fwd_iat_std",is_/1000)*1000
        rv=rs**2
        rx=row.get("fwd_iat_max",rm/1000)*1000; rn=row.get("fwd_iat_min",rm/1000)*1000
        # PacketLengthMode: most frequent size — approximate from fwd packets
        fwd_mode=row.get("fwd_pkt_len_min",pm)  # fwd min ≈ mode for DoH
        feat={
            "FlowBytesSent":row.get("totlen_fwd_pkts",0),
            "FlowBytesReceived":row.get("totlen_bwd_pkts",0),
            "FlowSentRate":row.get("flow_byts_s",0),
            "FlowReceivedRate":row.get("flow_pkts_s",0),
            "PacketLengthMean":pm,
            "PacketLengthVariance":pv,
            "PacketLengthStandardDeviation":ps,
            "PacketLengthMedian":(px+pn)/2,
            "PacketLengthMode":fwd_mode,  # Use fwd_min as mode proxy
            "PacketLengthSkewFromMedian":(pm-(px+pn)/2)/(ps+1e-9),
            "PacketLengthSkewFromMode":(pm-fwd_mode)/(ps+1e-9),
            "PacketLengthCoefficientofVariation":ps/(pm+1e-9),
            "PacketTimeMean":im,
            "PacketTimeVariance":iv,
            "PacketTimeStandardDeviation":is_,
            "PacketTimeMedian":(ix+ino)/2,
            "PacketTimeMode":im,
            "PacketTimeSkewFromMedian":(im-(ix+ino)/2)/(is_+1e-9),
            "PacketTimeSkewFromMode":0.0,
            "PacketTimeCoefficientofVariation":is_/(im+1e-9),
            "ResponseTimeTimeMean":rm,
            "ResponseTimeTimeVariance":rv,
            "ResponseTimeTimeStandardDeviation":rs,
            "ResponseTimeTimeMedian":(rx+rn)/2,
            "ResponseTimeTimeMode":rm,
            "ResponseTimeTimeSkewFromMedian":(rm-(rx+rn)/2)/(rs+1e-9),
            "ResponseTimeTimeSkewFromMode":0.0,
            "ResponseTimeTimeCoefficientofVariation":rs/(rm+1e-9),
        }
        rows.append([feat.get(f,0.0) for f in fn])
    return np.array(rows,dtype=np.float32) if rows else np.zeros((0,len(fn)))


# ── Validation: compare our features to benign ────────────────────────────────

def validate_features(df, dataset_path="data/l2-total-add.csv"):
    """Print comparison of our features vs real benign."""
    try:
        real=pd.read_csv(dataset_path,encoding='utf-8-sig')
        real.columns=[c.lstrip('\ufeff') for c in real.columns]
        benign=real[real['Label'].str.contains('Benign',case=False,na=False)]
    except:
        return

    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    mapping={
        'pkt_len_mean':'PacketLengthMean',
        'pkt_len_std':'PacketLengthStandardDeviation',
        'fwd_pkt_len_min':'PacketLengthMode',
        'flow_iat_std':'PacketTimeStandardDeviation',
        'totlen_fwd_pkts':'FlowBytesSent',
        'totlen_bwd_pkts':'FlowBytesReceived',
    }
    print(f"\n  {'Feature':<35} {'Ours':>10} {'Benign':>10} {'Match?':>8}")
    print(f"  {'-'*65}")
    for cic,det in mapping.items():
        if cic in df.columns and det in benign.columns:
            ours=df[cic].values[0]
            bm=benign[det].mean()
            bs=benign[det].std()
            ok="OK" if abs(ours-bm)<2*bs else "no"
            print(f"  {det:<35} {ours:>10.1f} {bm:>10.1f} {ok:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--flows", type=int, default=20)
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print feature comparison vs benign (first extracted flow per strategy)",
    )
    parser.add_argument(
        "--reference-data",
        default=None,
        help="L2 CSV/Parquet for NN interpolation + realism profile (default: auto-detect)",
    )
    parser.add_argument(
        "--target-sampling",
        choices=["interpolated", "correlated", "legacy"],
        default="interpolated",
        help=(
            "interpolated: kNN benign blend; correlated: MVN with Ledoit-Wolf covariance; "
            "legacy: v4 random wire stats"
        ),
    )
    parser.add_argument(
        "--correlation-validation",
        action="store_true",
        help=(
            "Compare Pearson correlation matrix of real benign CIRA rows vs many "
            "tabular samples from the active sampler; writes JSON report"
        ),
    )
    parser.add_argument(
        "--correlation-val-n",
        type=int,
        default=3000,
        metavar="N",
        help="Number of synthetic tabular rows for correlation validation",
    )
    parser.add_argument(
        "--correlation-report",
        default="correlation_validation_report.json",
        help="Output path for tabular correlation validation metrics",
    )
    parser.add_argument(
        "--realism-filter",
        action="store_true",
        help=(
            "Reject flows using mapped detector features vs CIRA benign marginals. "
            "CIC-derived rows rarely match raw CIRA p1-p99 on every dimension; use "
            "--realism-pass any and/or wider --realism-q-low/--realism-q-high."
        ),
    )
    parser.add_argument(
        "--realism-pass",
        choices=["all", "any"],
        default="any",
        help="all: both band and z counts must pass; any: pass if either passes (default, cross-domain)",
    )
    parser.add_argument(
        "--realism-q-low",
        type=float,
        default=0.01,
        help="Lower quantile for benign reference band (e.g. 0.05 loosens)",
    )
    parser.add_argument(
        "--realism-q-high",
        type=float,
        default=0.99,
        help="Upper quantile for benign reference band (e.g. 0.95 loosens)",
    )
    parser.add_argument(
        "--max-p99-violations",
        type=int,
        default=12,
        metavar="N",
        help="Max count of features outside the quantile band (name kept for CLI compatibility)",
    )
    parser.add_argument(
        "--max-z-violations",
        type=int,
        default=12,
        metavar="N",
        help="Max count of features with |z| > realism-z-max",
    )
    parser.add_argument("--realism-z-max", type=float, default=5.0)
    parser.add_argument(
        "--realism-retries",
        type=int,
        default=15,
        help="Attempts per scheduled flow when --realism-filter is set",
    )
    args = parser.parse_args()

    if not HAS_SCAPY:
        print("[!] pip install scapy")
        sys.exit(1)

    ref_path = resolve_reference_data(args.reference_data)
    realism_ref = (
        load_realism_profile(ref_path, args.realism_q_low, args.realism_q_high)
        if args.realism_filter
        else {}
    )

    sampler = None
    if args.target_sampling == "interpolated":
        try:
            sampler = InterpolatedBenignSampler(ref_path, seed=42)
        except Exception as e:
            print(f"[!] InterpolatedBenignSampler failed ({e}); using legacy targets")
            args.target_sampling = "legacy"
    elif args.target_sampling == "correlated":
        try:
            sampler = CorrelatedBenignSampler(ref_path, seed=42)
        except Exception as e:
            print(f"[!] CorrelatedBenignSampler failed ({e}); trying interpolated")
            args.target_sampling = "interpolated"
            try:
                sampler = InterpolatedBenignSampler(ref_path, seed=42)
            except Exception as e2:
                print(f"[!] InterpolatedBenignSampler failed ({e2}); using legacy targets")
                args.target_sampling = "legacy"
                sampler = None

    fn_path = os.path.join(args.results, "feature_names.json")
    fn = json.load(open(fn_path)) if os.path.exists(fn_path) else DETECTOR_FEATURES
    sc_path = os.path.join(args.results, "scaler.joblib")
    scaler = joblib.load(sc_path) if os.path.exists(sc_path) else None
    models = {}
    for n, f in [("RF", "rf.joblib"), ("GB", "gb.joblib"), ("XGB", "xgb.joblib")]:
        p = os.path.join(args.results, f)
        if os.path.exists(p):
            models[n] = joblib.load(p)
            print(f"[+] {n} loaded")

    shaper = TrafficShaper(args.fingerprint)
    payload = b"RESEARCH:" + b"X" * 503
    results = []
    tmpd = tempfile.mkdtemp(prefix="doh_adv_")

    print(f"\n{'='*65}")
    print("  REAL ADVERSARIAL PIPELINE v5 (interpolated targets + optional realism gate)")
    print(f"  Reference: {ref_path}")
    print(
        f"  target_sampling={args.target_sampling}  realism_filter={args.realism_filter}  "
        f"correlation_validation={args.correlation_validation}"
    )
    if args.realism_filter:
        print(
            f"  realism_pass={args.realism_pass}  q_band=[{args.realism_q_low},{args.realism_q_high}]  "
            f"max_band_violations={args.max_p99_violations}  max_z_violations={args.max_z_violations}  "
            f"z_max={args.realism_z_max}"
        )
    if args.realism_filter and args.max_p99_violations == 0 and args.max_z_violations == 0:
        print("  [*] Strict realism gate enabled: zero band and zero z violations allowed")
    print(f"{'='*65}")

    if args.correlation_validation:
        if sampler is None:
            print(
                "\n[!] --correlation-validation skipped (no tabular sampler; "
                "use --target-sampling interpolated or correlated)"
            )
        else:
            print(
                f"\n[*] Tabular correlation validation: N={args.correlation_val_n} "
                f"samples vs real benign ({type(sampler).__name__})"
            )
            rep = run_tabular_correlation_validation(
                ref_path,
                sampler,
                n_synthetic=args.correlation_val_n,
                report_path=args.correlation_report,
            )
            for key in sorted(k for k in rep if k != "feature_columns"):
                print(f"      {key}: {rep[key]}")
            print(f"      report: {args.correlation_report}")

    for strategy in EvasionStrategy:
        print(f"\n-- {strategy.value} --------------------------------------")
        vecs = []
        did_validate = False
        stats = {
            "accepted": 0,
            "empty_cic": 0,
            "empty_map": 0,
            "rejected_realism": 0,
            "best_p99v": None,
            "best_zv": None,
        }

        for i in range(args.flows):
            sched = shaper.build_schedule(payload, strategy=strategy, chunk_size=40)
            attempts = args.realism_retries if args.realism_filter else 1
            m_accept = None

            for _ in range(attempts):
                if (
                    args.target_sampling in ("interpolated", "correlated")
                    and sampler is not None
                ):
                    tfeat = sampler.sample()
                    tp = traffic_params_from_cira_features(tfeat)
                else:
                    tp = None

                pkts = schedule_to_session(
                    sched, start_time=time.time() + i * 30.0, traffic_params=tp
                )
                if not pkts:
                    break

                pcap = os.path.join(tmpd, f"adv_{strategy.value}_{i}.pcap")
                csv = os.path.join(tmpd, f"adv_{strategy.value}_{i}.csv")
                wrpcap(pcap, pkts)
                df = run_cicflowmeter(pcap, csv)
                for path in (pcap, csv):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

                if df.empty:
                    stats["empty_cic"] += 1
                    continue

                if "dst_ip" in df.columns:
                    c2 = df[df["dst_ip"].astype(str).str.contains("104.16", na=False)]
                    df_use = c2 if not c2.empty else df
                else:
                    df_use = df

                if i == 0 and args.validate and not did_validate and not df_use.empty:
                    validate_features(df_use.copy())
                    did_validate = True

                m = map_features(df_use, fn)
                if len(m) == 0:
                    stats["empty_map"] += 1
                    continue

                row = row_dict_from_matrix(m, fn)
                if not args.realism_filter or not realism_ref:
                    m_accept = m
                    stats["accepted"] += 1
                    break
                band_v, zv, _ = score_realism(row, realism_ref, z_max=args.realism_z_max)
                if stats["best_p99v"] is None or band_v < stats["best_p99v"]:
                    stats["best_p99v"] = band_v
                if stats["best_zv"] is None or zv < stats["best_zv"]:
                    stats["best_zv"] = zv
                ok_band = band_v <= args.max_p99_violations
                ok_z = zv <= args.max_z_violations
                if args.realism_pass == "any":
                    ok = ok_band or ok_z
                else:
                    ok = ok_band and ok_z
                if ok:
                    m_accept = m
                    stats["accepted"] += 1
                    break
                stats["rejected_realism"] += 1

            if m_accept is not None and len(m_accept) > 0:
                vecs.append(m_accept[0])

        if not vecs:
            print("  [!] No features extracted")
            if args.realism_filter:
                print(
                    "      Diagnostics: "
                    f"accepted={stats['accepted']}, "
                    f"rejected_realism={stats['rejected_realism']}, "
                    f"empty_cic={stats['empty_cic']}, "
                    f"empty_map={stats['empty_map']}, "
                    f"best_band_v={stats['best_p99v']}, "
                    f"best_zv={stats['best_zv']}  "
                    f"(try --realism-pass any, wider --realism-q-low/--realism-q-high, "
                    f"or omit --realism-filter)"
                )
            continue

        X = np.array(vecs)
        if scaler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = scaler.transform(X)

        for mn, model in models.items():
            probs = model.predict_proba(X)[:, 1]
            det = int((probs > 0.5).sum())
            ev = 1 - det / len(probs)
            print(
                f"   {mn:6s} -> evasion={ev:.1%}  conf={probs.mean():.3f}  n={len(probs)}"
            )
            results.append(
                {
                    "strategy": strategy.value,
                    "model": mn,
                    "evasion_rate": ev,
                    "mean_confidence": float(probs.mean()),
                    "n_flows": len(probs),
                }
            )

    try:
        os.rmdir(tmpd)
    except OSError:
        pass

    df_out = pd.DataFrame(results)
    df_out.to_csv("real_adversarial_results.csv", index=False)
    df_out.to_json("real_adversarial_results.json", orient="records", indent=2)
    print("\n[+] Saved to real_adversarial_results.csv")

    if not df_out.empty:
        print(f"\n{'='*65}")
        print("  RESULTS SUMMARY")
        print(f"{'='*65}")
        pivot = df_out.pivot_table(
            index="strategy", columns="model", values="evasion_rate"
        )
        order = [s.value for s in EvasionStrategy]
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        print(pivot.map(lambda x: f"{x:.1%}").to_string())


if __name__ == "__main__":
    main()
