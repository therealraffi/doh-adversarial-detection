"""
export_adversarial_dataset.py
==============================
Generates adversarial DoH C2 flows and exports them as a labeled CSV dataset.
Mughesh can use this as hard negatives for retraining the detectors.

Output: adversarial_flows_dataset.csv
  - Same 28 features as the training data
  - Label column: "Adversarial" (so it can be added to training set)
  - Strategy column: which evasion strategy generated it
  - Attack_type: "whitebox" or "blackbox" (for metadata)

Usage:
    python export_adversarial_dataset.py --results ./results_full --flows 50
"""

import os, sys, time, random, subprocess, json, argparse, warnings
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_shaper import TrafficShaper, EvasionStrategy

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
except ImportError:
    print("[!] pip install scapy"); sys.exit(1)

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

# ── CIRA-CIC benign distributions ─────────────────────────────────────────────
_STATS = None

def _load_stats(path="data/l2-total-add.csv"):
    global _STATS
    if _STATS: return _STATS
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        df.dropna(subset=["PacketLengthMode","PacketLengthMean",
                           "PacketTimeMean","FlowBytesSent"], inplace=True)
        b = df[df["Label"].str.contains("Benign", case=False, na=False)]
        _STATS = {
            "pkt_mode_mean": float(b["PacketLengthMode"].mean()),
            "pkt_mode_std":  float(b["PacketLengthMode"].std()),
            "pkt_mean_mean": float(b["PacketLengthMean"].mean()),
            "pkt_mean_std":  float(b["PacketLengthMean"].std()),
            "iat_mean":      float(b["PacketTimeMean"].mean()),
            "iat_std":       float(b["PacketTimeStandardDeviation"].mean()),
            "n_pkt_mean":    float(b["FlowBytesSent"].mean() / b["PacketLengthMean"].mean()),
        }
        print(f"[+] Loaded CIRA-CIC distributions ({len(b)} benign flows)")
    except Exception as e:
        print(f"[!] Using literature values ({e})")
        _STATS = {"pkt_mode_mean":74.1,"pkt_mode_std":20.3,
                  "pkt_mean_mean":137.5,"pkt_mean_std":81.8,
                  "iat_mean":45.0,"iat_std":11.0,"n_pkt_mean":455}
    return _STATS

def sample_pkt():
    s = _load_stats()
    return max(40, int(random.gauss(s["pkt_mode_mean"], s["pkt_mode_std"]))) \
           if random.random() < 0.6 else \
           max(100, int(random.gauss(s["pkt_mean_mean"]*1.5, s["pkt_mean_std"])))

def sample_iat():
    s = _load_stats()
    return max(5, random.gauss(s["iat_mean"], s["iat_std"]))

def sample_n():
    s = _load_stats()
    return max(50, int(random.gauss(s["n_pkt_mean"], s["n_pkt_mean"]*0.3)))


# ── TCP session ───────────────────────────────────────────────────────────────

def tcp_hs(sm,dm,si,di,sp,dp,t):
    p=[]
    for flags,src,dst,sip,dip in [('S',sm,dm,si,di),('SA',dm,sm,di,si),('A',sm,dm,si,di)]:
        pk=Ether(src=src,dst=dst)/IP(src=sip,dst=dip)/TCP(sport=sp,dport=dp,flags=flags,seq=1000)
        pk.time=t; p.append(pk); t+=random.uniform(0.001,0.005)
    return p, t, 1001, 2001

def tcp_td(sm,dm,si,di,sp,dp,t,sc,ss):
    p=[]
    f=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='FA',seq=sc,ack=ss)
    f.time=t; p.append(f); t+=0.002
    fa=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/TCP(sport=dp,dport=sp,flags='FA',seq=ss,ack=sc+1)
    fa.time=t; p.append(fa); t+=0.002
    a=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='A',seq=sc+1,ack=ss+1)
    a.time=t; p.append(a)
    return p

def build_session(schedule, t0):
    sm="00:1A:2B:3C:4D:5E"; si="192.168.1.50"
    sp=random.randint(1024,65535); dp=443
    di="104.16.248.249"; dm="AA:BB:CC:DD:EE:11"
    pkts=[]; t=t0
    real_q=[q for q in schedule if not q.is_cover]
    hs,t,sc,ss=tcp_hs(sm,dm,si,di,sp,dp,t)
    pkts.extend(hs)
    n=max(len(real_q), sample_n())
    for i in range(n):
        t+=(real_q[i].delay_ms/1000.0 if i<len(real_q) else sample_iat()/1000.0)
        rs=sample_pkt()
        req=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/TCP(sport=sp,dport=dp,flags='PA',seq=sc,ack=ss)/Raw(b"E"*max(1,rs-54))
        req.time=t; pkts.append(req); sc+=rs
        t+=sample_iat()/1000.0
        rrs=max(60,int(random.gauss(200,80)))
        resp=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/TCP(sport=dp,dport=sp,flags='PA',seq=ss,ack=sc)/Raw(b"V"*max(1,rrs-54))
        resp.time=t; pkts.append(resp); ss+=rrs
    pkts.extend(tcp_td(sm,dm,si,di,sp,dp,t,sc,ss))
    pkts.sort(key=lambda p: p.time)
    return pkts


# ── CICFlowMeter + feature mapping ───────────────────────────────────────────

def run_cic(pcap, csv):
    try:
        subprocess.run(["cicflowmeter","-f",pcap,"-c",csv],
                       capture_output=True,timeout=60)
    except Exception as e:
        print(f"[!] {e}"); return pd.DataFrame()
    if not os.path.exists(csv): return pd.DataFrame()
    df=pd.read_csv(csv)
    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    return df

def extract_features(df):
    rows=[]
    for _,row in df.iterrows():
        pm=row.get("pkt_len_mean",0); ps=row.get("pkt_len_std",0)
        pv=ps**2; px=row.get("pkt_len_max",pm); pn=row.get("pkt_len_min",pm)
        im=row.get("flow_iat_mean",0)*1000; is_=row.get("flow_iat_std",0)*1000
        iv=is_**2; ix=row.get("flow_iat_max",0)*1000; ino=row.get("flow_iat_min",0)*1000
        rm=row.get("fwd_iat_mean",0)*1000; rs=row.get("fwd_iat_std",0)*1000
        rv=rs**2; rx=row.get("fwd_iat_max",0)*1000; rn=row.get("fwd_iat_min",0)*1000
        fm=row.get("fwd_pkt_len_min",pm)
        feat={
            "FlowBytesSent":row.get("totlen_fwd_pkts",0),
            "FlowBytesReceived":row.get("totlen_bwd_pkts",0),
            "FlowSentRate":row.get("flow_byts_s",0),
            "FlowReceivedRate":row.get("flow_pkts_s",0),
            "PacketLengthMean":pm,"PacketLengthVariance":pv,
            "PacketLengthStandardDeviation":ps,
            "PacketLengthMedian":(px+pn)/2,"PacketLengthMode":fm,
            "PacketLengthSkewFromMedian":(pm-(px+pn)/2)/(ps+1e-9),
            "PacketLengthSkewFromMode":(pm-fm)/(ps+1e-9),
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
        rows.append([feat.get(f,0.0) for f in DETECTOR_FEATURES])
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--flows",   type=int, default=50,
                        help="Flows per strategy (default 50, total = 50 x 6 = 300)")
    parser.add_argument("--output",  default="adversarial_flows_dataset.csv")
    args=parser.parse_args()

    shaper  = TrafficShaper()
    payload = b"RESEARCH:" + b"X"*503
    all_rows = []

    print(f"\n{'='*60}")
    print("  ADVERSARIAL DATASET EXPORT")
    print(f"  {args.flows} flows per strategy x 6 strategies = {args.flows*6} total rows")
    print(f"{'='*60}\n")

    for strategy in EvasionStrategy:
        print(f"Generating {strategy.value}...")
        count = 0
        for i in range(args.flows):
            sched = shaper.build_schedule(payload, strategy=strategy, chunk_size=40)
            pkts  = build_session(sched, time.time()+i*30.0)
            if not pkts: continue
            pcap = f"/tmp/exp_{strategy.value}_{i}.pcap"
            csv  = f"/tmp/exp_{strategy.value}_{i}.csv"
            wrpcap(pcap, pkts)
            df = run_cic(pcap, csv)
            if df.empty: continue
            if "dst_ip" in df.columns:
                c2 = df[df["dst_ip"].astype(str).str.contains("104.16", na=False)]
                df = c2 if not c2.empty else df
            rows = extract_features(df)
            for row in rows:
                all_rows.append(row + [strategy.value, "Adversarial"])
                count += 1
            for f in [pcap, csv]:
                try: os.remove(f)
                except: pass
        print(f"  {count} flows extracted")

    # Build final dataframe
    cols = DETECTOR_FEATURES + ["Strategy", "Label"]
    df_out = pd.DataFrame(all_rows, columns=cols)

    df_out.to_csv(args.output, index=False)
    print(f"\n[+] Saved {len(df_out)} adversarial flows to {args.output}")
    print(f"    Columns: {len(DETECTOR_FEATURES)} features + Strategy + Label")
    print(f"\n    Strategy breakdown:")
    print(df_out["Strategy"].value_counts().to_string())
    print(f"\n    To use for retraining, concat with training data:")
    print(f"    df_train = pd.concat([df_original, df_adversarial])")
    print(f"    where df_adversarial = pd.read_csv('{args.output}')")

if __name__=="__main__":
    main()
