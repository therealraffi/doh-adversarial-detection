"""
blackbox_adversarial_pipeline.py
=================================
REALISTIC ATTACKER VERSION — only uses what a real attacker has access to:

  HAVE:
    - Observed benign DoH traffic (CIRA-CIC browser flows)
    - CICFlowMeter (public tool)
    - Binary feedback: C2 blocked or not blocked
    - Scapy (public tool)

  DO NOT HAVE:
    - The trained model internals
    - The scaler (scaler.joblib)
    - The exact 28 feature names the model uses
    - SHAP values or feature importance
    - predict_proba() — only binary blocked/not blocked

This script estimates what the attacker sees in the real world.
Compare results with real_adversarial_pipeline.py (white-box upper bound).

Usage:
    python blackbox_adversarial_pipeline.py --results ./results_full --flows 20
"""

import os, sys, time, random, subprocess, json, argparse, warnings
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_shaper import TrafficShaper, EvasionStrategy

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("[!] pip install scapy")

# ── What the attacker knows ───────────────────────────────────────────────────
# The attacker observes benign DoH traffic on their own network.
# They use CICFlowMeter (public) to extract features from their observed traffic.
# They do NOT know which features the defender uses or how they are scaled.

_CIRA_STATS = None

def _load_cira_stats(dataset_path="data/l2-total-add.csv"):
    """
    Load benign distributions from CIRA-CIC browser flows.
    This simulates the attacker observing real benign DoH traffic.
    Independent of the Hokkaido malicious training signal.
    """
    global _CIRA_STATS
    if _CIRA_STATS is not None:
        return _CIRA_STATS
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        df.dropna(subset=["PacketLengthMode","PacketLengthMean","PacketTimeMean",
                           "PacketTimeStandardDeviation","FlowBytesSent"], inplace=True)
        benign = df[df["Label"].str.contains("Benign", case=False, na=False)]
        n_pkt = float(benign["FlowBytesSent"].mean() / benign["PacketLengthMean"].mean())
        _CIRA_STATS = {
            "pkt_mode_mean": float(benign["PacketLengthMode"].mean()),
            "pkt_mode_std":  float(benign["PacketLengthMode"].std()),
            "pkt_mean_mean": float(benign["PacketLengthMean"].mean()),
            "pkt_mean_std":  float(benign["PacketLengthMean"].std()),
            "iat_mean":      float(benign["PacketTimeMean"].mean()),
            "iat_std":       float(benign["PacketTimeStandardDeviation"].mean()),
            "n_pkt_mean":    n_pkt,
        }
        print(f"[+] Attacker observed {len(benign)} benign CIRA-CIC flows")
        print(f"    Packet size mode: {_CIRA_STATS['pkt_mode_mean']:.1f}B")
        print(f"    Avg flow length:  {_CIRA_STATS['n_pkt_mean']:.0f} packets")
    except Exception as e:
        print(f"[!] Could not load CIRA-CIC: {e}. Using literature values.")
        _CIRA_STATS = {
            "pkt_mode_mean": 74.1, "pkt_mode_std": 20.3,
            "pkt_mean_mean": 137.5, "pkt_mean_std": 81.8,
            "iat_mean": 45.0, "iat_std": 11.0, "n_pkt_mean": 455,
        }
    return _CIRA_STATS


def sample_packet_size():
    s = _load_cira_stats()
    if random.random() < 0.6:
        return max(40, int(random.gauss(s["pkt_mode_mean"], s["pkt_mode_std"])))
    else:
        return max(100, int(random.gauss(s["pkt_mean_mean"]*1.5, s["pkt_mean_std"])))

def sample_iat_ms():
    s = _load_cira_stats()
    return max(5, random.gauss(s["iat_mean"], s["iat_std"]))

def sample_n_packets():
    s = _load_cira_stats()
    return max(50, int(random.gauss(s["n_pkt_mean"], s["n_pkt_mean"]*0.3)))


# ── TCP session builder ───────────────────────────────────────────────────────

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


def schedule_to_session(schedule, start_time):
    sm="00:1A:2B:3C:4D:5E"; si="192.168.1.50"
    sp=random.randint(1024,65535); dp=443
    all_pkts=[]
    real_q=[q for q in schedule if not q.is_cover]
    cover_q=[q for q in schedule if q.is_cover]

    if real_q:
        di="104.16.248.249"; dm="AA:BB:CC:DD:EE:11"
        t=start_time
        hs,t,sc,ss=tcp_handshake(sm,dm,si,di,sp,dp,t)
        all_pkts.extend(hs)
        n_packets=max(len(real_q), sample_n_packets())
        for i in range(n_packets):
            t += (real_q[i].delay_ms/1000.0 if i < len(real_q) else sample_iat_ms()/1000.0)
            rs=sample_packet_size()
            req=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='PA',seq=sc,ack=ss)/Raw(b"E"*max(1,rs-54))
            req.time=t; all_pkts.append(req); sc+=rs
            t+=sample_iat_ms()/1000.0
            rrs=max(60,int(random.gauss(200,80)))
            resp=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/TCP(sport=dp,dport=sp,flags='PA',seq=ss,ack=sc)/Raw(b"V"*max(1,rrs-54))
            resp.time=t; all_pkts.append(resp); ss+=rrs
        all_pkts.extend(tcp_teardown(sm,dm,si,di,sp,dp,t,sc,ss))

    for q in cover_q:
        cdi="8.8.8.8"; cdm="AA:BB:CC:DD:EE:00"
        csp=random.randint(1024,65535)
        ct=start_time+q.delay_ms/1000.0
        hs,ct,sc2,ss2=tcp_handshake(sm,cdm,si,cdi,csp,dp,ct)
        all_pkts.extend(hs)
        for _ in range(sample_n_packets()):
            ct+=sample_iat_ms()/1000.0
            rs=sample_packet_size()
            req=Ether(src=sm,dst=cdm)/IP(src=si,dst=cdi)/TCP(sport=csp,dport=dp,flags='PA',seq=sc2,ack=ss2)/Raw(b"B"*max(1,rs-54))
            req.time=ct; all_pkts.append(req); sc2+=rs
            ct+=sample_iat_ms()/1000.0
            rrs=max(60,int(random.gauss(200,80)))
            resp=Ether(src=cdm,dst=sm)/IP(src=cdi,dst=si)/TCP(sport=dp,dport=csp,flags='PA',seq=ss2,ack=sc2)/Raw(b"G"*max(1,rrs-54))
            resp.time=ct; all_pkts.append(resp); ss2+=rrs
        all_pkts.extend(tcp_teardown(sm,cdm,si,cdi,csp,dp,ct,sc2,ss2))

    all_pkts.sort(key=lambda p: p.time)
    return all_pkts


# ── CICFlowMeter ─────────────────────────────────────────────────────────────

def run_cicflowmeter(pcap, csv_out):
    try:
        subprocess.run(["cicflowmeter","-f",pcap,"-c",csv_out],
                       capture_output=True,timeout=60)
    except Exception as e:
        print(f"[!] {e}"); return pd.DataFrame()
    if not os.path.exists(csv_out): return pd.DataFrame()
    df=pd.read_csv(csv_out)
    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    return df


# ── Black-box feature extraction ──────────────────────────────────────────────
# The attacker uses ALL CICFlowMeter columns directly.
# No knowledge of which 28 features the model uses.
# No scaler applied.

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

def map_features_blackbox(df, feature_names):
    """
    Map CICFlowMeter columns to detector features.
    NO SCALER APPLIED — attacker doesn't have scaler.joblib.
    Time values multiplied by 1000 (seconds → milliseconds)
    because CICFlowMeter outputs seconds but training data used ms.
    NOTE: The attacker could figure this unit mismatch out by
    comparing CICFlowMeter output to published CIRA-CIC feature ranges.
    """
    rows=[]
    for _,row in df.iterrows():
        pm=row.get("pkt_len_mean",row.get("pkt_size_avg",0))
        ps=row.get("pkt_len_std",0)
        pv=ps**2
        px=row.get("pkt_len_max",pm); pn=row.get("pkt_len_min",pm)
        # Attacker figures out ms conversion from published dataset stats
        im=row.get("flow_iat_mean",0)*1000
        is_=row.get("flow_iat_std",0)*1000
        iv=is_**2
        ix=row.get("flow_iat_max",0)*1000; ino=row.get("flow_iat_min",0)*1000
        rm=row.get("fwd_iat_mean",0)*1000
        rs=row.get("fwd_iat_std",0)*1000
        rv=rs**2
        rx=row.get("fwd_iat_max",0)*1000; rn=row.get("fwd_iat_min",0)*1000
        fwd_mode=row.get("fwd_pkt_len_min",pm)
        feat={
            "FlowBytesSent":row.get("totlen_fwd_pkts",0),
            "FlowBytesReceived":row.get("totlen_bwd_pkts",0),
            "FlowSentRate":row.get("flow_byts_s",0),
            "FlowReceivedRate":row.get("flow_pkts_s",0),
            "PacketLengthMean":pm,"PacketLengthVariance":pv,
            "PacketLengthStandardDeviation":ps,
            "PacketLengthMedian":(px+pn)/2,
            "PacketLengthMode":fwd_mode,
            "PacketLengthSkewFromMedian":(pm-(px+pn)/2)/(ps+1e-9),
            "PacketLengthSkewFromMode":(pm-fwd_mode)/(ps+1e-9),
            "PacketLengthCoefficientofVariation":ps/(pm+1e-9),
            "PacketTimeMean":im,"PacketTimeVariance":iv,
            "PacketTimeStandardDeviation":is_,
            "PacketTimeMedian":(ix+ino)/2,"PacketTimeMode":im,
            "PacketTimeSkewFromMedian":(im-(ix+ino)/2)/(is_+1e-9),
            "PacketTimeSkewFromMode":0.0,
            "PacketTimeCoefficientofVariation":is_/(im+1e-9),
            "ResponseTimeTimeMean":rm,"ResponseTimeTimeVariance":rv,
            "ResponseTimeTimeStandardDeviation":rs,
            "ResponseTimeTimeMedian":(rx+rn)/2,"ResponseTimeTimeMode":rm,
            "ResponseTimeTimeSkewFromMedian":(rm-(rx+rn)/2)/(rs+1e-9),
            "ResponseTimeTimeSkewFromMode":0.0,
            "ResponseTimeTimeCoefficientofVariation":rs/(rm+1e-9),
        }
        rows.append([feat.get(f,0.0) for f in feature_names])
    return np.array(rows,dtype=np.float32) if rows else np.zeros((0,len(feature_names)))


# ── Black-box classifier interface ───────────────────────────────────────────
# Attacker gets BINARY feedback only: blocked (1) or not blocked (0).
# No access to predict_proba() confidence scores.
# No scaler applied.

class BlackBoxDetector:
    """
    Simulates what a real attacker sees:
    - Can send traffic and observe if it gets blocked
    - Cannot access model internals, scaler, or confidence scores
    - Only binary feedback
    """
    def __init__(self, results_dir):
        self.models = {}
        self.feature_names = DETECTOR_FEATURES  # attacker guesses from CICFlowMeter output
        # NOTE: No scaler loaded — attacker doesn't have it

        for name, fname in [("RF","rf.joblib"),("GB","gb.joblib"),("XGB","xgb.joblib")]:
            p = os.path.join(results_dir, fname)
            if os.path.exists(p):
                self.models[name] = joblib.load(p)
                print(f"[+] {name} loaded (black-box — no scaler)")

    def is_blocked(self, feature_vec, model_name):
        """
        Binary detection: returns True if detected (blocked), False if evaded.
        No scaler, no confidence — just the raw model prediction.
        """
        model = self.models.get(model_name)
        if model is None:
            return False
        X = feature_vec.reshape(1, -1)
        # NO scaler.transform() — attacker doesn't have it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(X)[0]
        return bool(pred == 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Black-box adversarial pipeline — realistic attacker model"
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--flows",   type=int, default=20)
    args = parser.parse_args()

    detector = BlackBoxDetector(args.results)
    shaper   = TrafficShaper()
    payload  = b"RESEARCH:" + b"X" * 503
    results  = []

    print(f"\n{'='*65}")
    print("  BLACK-BOX ADVERSARIAL PIPELINE")
    print("  Realistic attacker: no scaler, no feature names, binary feedback only")
    print(f"{'='*65}")
    print(f"\n  Attacker knowledge:")
    print(f"    ✓ Observed benign DoH traffic (CIRA-CIC browser flows)")
    print(f"    ✓ CICFlowMeter (public tool)")
    print(f"    ✓ Scapy (public tool)")
    print(f"    ✗ Model scaler (scaler.joblib)")
    print(f"    ✗ Exact feature names used by detector")
    print(f"    ✗ SHAP / feature importance")
    print(f"    ✗ predict_proba() — binary feedback only")

    for strategy in EvasionStrategy:
        print(f"\n── {strategy.value} ──────────────────────")
        detected_by  = {m: 0 for m in detector.models}
        total_flows  = 0

        for i in range(args.flows):
            sched = shaper.build_schedule(payload, strategy=strategy, chunk_size=40)
            pkts  = schedule_to_session(sched, start_time=time.time()+i*30.0)
            if not pkts: continue

            pcap = f"/tmp/bb_{strategy.value}_{i}.pcap"
            csv  = f"/tmp/bb_{strategy.value}_{i}.csv"
            wrpcap(pcap, pkts)
            df = run_cicflowmeter(pcap, csv)
            if df.empty: continue

            if "dst_ip" in df.columns:
                c2 = df[df["dst_ip"].astype(str).str.contains("104.16", na=False)]
                df_use = c2 if not c2.empty else df
            else:
                df_use = df

            m = map_features_blackbox(df_use, DETECTOR_FEATURES)
            if len(m) == 0: continue

            total_flows += 1
            for mn in detector.models:
                if detector.is_blocked(m[0], mn):
                    detected_by[mn] += 1

            for f in [pcap, csv]:
                try: os.remove(f)
                except: pass

        if total_flows == 0:
            print("  [!] No flows processed"); continue

        for mn in detector.models:
            ev = 1 - detected_by[mn] / total_flows
            print(f"   {mn:6s} → evasion={ev:.1%}  n={total_flows}")
            results.append({
                "strategy":     strategy.value,
                "model":        mn,
                "evasion_rate": ev,
                "n_flows":      total_flows,
                "attack_type":  "black-box",
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv("blackbox_results.csv", index=False)
    df_out.to_json("blackbox_results.json", orient="records", indent=2)
    print(f"\n[+] Saved to blackbox_results.csv")

    if not df_out.empty:
        print(f"\n{'='*65}")
        print("  BLACK-BOX RESULTS SUMMARY")
        print(f"{'='*65}")
        pivot = df_out.pivot_table(
            index="strategy", columns="model", values="evasion_rate"
        )
        order = [s.value for s in EvasionStrategy]
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        print(pivot.map(lambda x: f"{x:.1%}").to_string())

        print(f"\n{'='*65}")
        print("  COMPARE WITH WHITE-BOX (real_adversarial_pipeline.py)")
        print("  White-box uses scaler + predict_proba + known feature names")
        print("  Black-box uses raw unscaled features + binary feedback only")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
