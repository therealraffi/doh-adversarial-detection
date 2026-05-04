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

# ── Real benign distributions (from actual dataset) ────────────────────────────
# PacketLengthMode   mean=74,  std=20,  p5=54,   p95=105
# PacketLengthMean   mean=137, std=82,  p5=73,   p95=254
# PacketLengthMedian mean=94,  std=17,  p5=66,   p95=123
# PacketTimeStd      mean=11,  std=18,  p5=0,    p95=58
# FlowBytesSent      mean=10KB,std=30KB,p5=128B, p95=68KB

# CIRA-CIC benign distributions (loaded at runtime)
# INDEPENDENT of the Hokkaido malicious tunnel training signal.
_CIRA_STATS = None

def _load_cira_stats(dataset_path='data/l2-total-add.csv'):
    global _CIRA_STATS
    if _CIRA_STATS is not None:
        return _CIRA_STATS
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        df.columns = [c.lstrip('\ufeff') for c in df.columns]
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df.dropna(subset=['PacketLengthMode','PacketLengthMean','PacketTimeMean',
                          'PacketTimeStandardDeviation','FlowBytesSent'], inplace=True)
        benign = df[df['Label'].str.contains('Benign', case=False, na=False)]
        n_pkt = float(benign['FlowBytesSent'].mean() / benign['PacketLengthMean'].mean())
        _CIRA_STATS = {
            'pkt_mode_mean': float(benign['PacketLengthMode'].mean()),
            'pkt_mode_std':  float(benign['PacketLengthMode'].std()),
            'pkt_mean_mean': float(benign['PacketLengthMean'].mean()),
            'pkt_mean_std':  float(benign['PacketLengthMean'].std()),
            'iat_mean':      float(benign['PacketTimeMean'].mean()),
            'iat_std':       float(benign['PacketTimeStandardDeviation'].mean()),
            'n_pkt_mean':    n_pkt,
        }
        print(f'[+] Gray-box: CIRA-CIC benign loaded ({len(benign)} flows) - independent of malicious training')
        print(f"    PacketLengthMode: {_CIRA_STATS['pkt_mode_mean']:.1f}B")
        print(f"    PacketLengthMean: {_CIRA_STATS['pkt_mean_mean']:.1f}B")
        print(f"    Avg packets/flow: {_CIRA_STATS['n_pkt_mean']:.0f}")
    except Exception as e:
        print(f'[!] CIRA-CIC load failed: {e}. Using literature values.')
        _CIRA_STATS = {
            'pkt_mode_mean': 74.1, 'pkt_mode_std': 20.3,
            'pkt_mean_mean': 137.5, 'pkt_mean_std': 81.8,
            'iat_mean': 45.0, 'iat_std': 11.0, 'n_pkt_mean': 455,
        }
    return _CIRA_STATS

def sample_packet_size():
    s = _load_cira_stats()
    if __import__('random').random() < 0.6:
        return max(40, int(__import__('random').gauss(s['pkt_mode_mean'], s['pkt_mode_std'])))
    else:
        return max(100, int(__import__('random').gauss(s['pkt_mean_mean']*1.5, s['pkt_mean_std'])))

def sample_iat_ms():
    s = _load_cira_stats()
    return max(5, __import__('random').gauss(s['iat_mean'], s['iat_std']))

def sample_n_packets():
    s = _load_cira_stats()
    return max(50, int(__import__('random').gauss(s['n_pkt_mean'], s['n_pkt_mean']*0.3)))



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
    """
    Build TCP session with real benign size/timing distributions.
    IAT comes from our evasion shaper.
    Packet sizes match real benign DoH (74B mode).
    """
    sm="00:1A:2B:3C:4D:5E"; si="192.168.1.50"
    sp=random.randint(1024,65535); dp=443
    all_pkts=[]

    real_q = [q for q in schedule if not q.is_cover]
    cover_q = [q for q in schedule if q.is_cover]

    # ── C2 session ────────────────────────────────────────────────────────────
    if real_q:
        di="104.16.248.249"; dm="AA:BB:CC:DD:EE:11"
        t=start_time

        hs,t,sc,ss=tcp_handshake(sm,dm,si,di,sp,dp,t)
        all_pkts.extend(hs)

        # Use more packets to match FlowBytesSent distribution
        n_packets = max(len(real_q), random.randint(400, 600))

        for i in range(n_packets):
            if i < len(real_q):
                # Use shaper IAT for real queries
                t += real_q[i].delay_ms / 1000.0
            else:
                # Pad with benign-timed packets
                t += sample_iat_ms() / 1000.0

            # Benign-sized query (~74B)
            req_size = sample_packet_size()
            req=Ether(src=sm,dst=dm)/IP(src=si,dst=di)/\
                TCP(sport=sp,dport=dp,flags='PA',seq=sc,ack=ss)/\
                Raw(b"E"*max(1,req_size-54))  # subtract TCP/IP overhead
            req.time=t; all_pkts.append(req); sc+=req_size

            # Response: slightly larger
            t+=sample_iat_ms()/1000.0
            resp_size=max(60, int(random.gauss(200, 80)))
            resp=Ether(src=dm,dst=sm)/IP(src=di,dst=si)/\
                 TCP(sport=dp,dport=sp,flags='PA',seq=ss,ack=sc)/\
                 Raw(b"V"*max(1,resp_size-54))
            resp.time=t; all_pkts.append(resp); ss+=resp_size

        td=tcp_teardown(sm,dm,si,di,sp,dp,t,sc,ss)
        all_pkts.extend(td)

    # ── Cover sessions ────────────────────────────────────────────────────────
    for q in cover_q:
        cdi="8.8.8.8"; cdm="AA:BB:CC:DD:EE:00"
        csp=random.randint(1024,65535)
        ct=start_time+q.delay_ms/1000.0
        hs,ct,sc2,ss2=tcp_handshake(sm,cdm,si,cdi,csp,dp,ct)
        all_pkts.extend(hs)

        n_cover=sample_n_packets()
        for _ in range(n_cover):
            ct+=sample_iat_ms()/1000.0
            rs=sample_packet_size()
            req=Ether(src=sm,dst=cdm)/IP(src=si,dst=cdi)/\
                TCP(sport=csp,dport=dp,flags='PA',seq=sc2,ack=ss2)/\
                Raw(b"B"*max(1,rs-54))
            req.time=ct; all_pkts.append(req); sc2+=rs
            ct+=sample_iat_ms()/1000.0
            rrs=max(60,int(random.gauss(200,80)))
            resp=Ether(src=cdm,dst=sm)/IP(src=cdi,dst=si)/\
                 TCP(sport=dp,dport=csp,flags='PA',seq=ss2,ack=sc2)/\
                 Raw(b"G"*max(1,rrs-54))
            resp.time=ct; all_pkts.append(resp); ss2+=rrs

        td=tcp_teardown(sm,cdm,si,cdi,csp,dp,ct,sc2,ss2)
        all_pkts.extend(td)

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
            ok="✓" if abs(ours-bm)<2*bs else "✗"
            print(f"  {det:<35} {ours:>10.1f} {bm:>10.1f} {ok:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",required=True)
    parser.add_argument("--flows",type=int,default=20)
    parser.add_argument("--fingerprint",default=None)
    parser.add_argument("--validate",action="store_true",
                        help="Print feature comparison vs benign")
    args=parser.parse_args()

    fn_path=os.path.join(args.results,"feature_names.json")
    fn=json.load(open(fn_path)) if os.path.exists(fn_path) else DETECTOR_FEATURES
    sc_path=os.path.join(args.results,"scaler.joblib")
    scaler=joblib.load(sc_path) if os.path.exists(sc_path) else None
    models={}
    for n,f in [("RF","rf.joblib"),("GB","gb.joblib"),("XGB","xgb.joblib")]:
        p=os.path.join(args.results,f)
        if os.path.exists(p): models[n]=joblib.load(p); print(f"[+] {n} loaded")

    shaper=TrafficShaper(args.fingerprint)
    payload=b"RESEARCH:"+b"X"*503
    results=[]

    print(f"\n{'='*65}")
    print("  REAL ADVERSARIAL PIPELINE v4 — Real Distribution Targeting")
    print(f"{'='*65}")

    for strategy in EvasionStrategy:
        print(f"\n── {strategy.value} ──────────────────────")
        vecs=[]

        for i in range(args.flows):
            sched=shaper.build_schedule(payload,strategy=strategy,chunk_size=40)
            pkts=schedule_to_session(sched,start_time=time.time()+i*30.0)
            if not pkts: continue
            pcap=f"/tmp/adv_{strategy.value}_{i}.pcap"
            csv=f"/tmp/adv_{strategy.value}_{i}.csv"
            wrpcap(pcap,pkts)
            df=run_cicflowmeter(pcap,csv)
            if df.empty: continue

            # Validate first flow
            if i==0 and args.validate:
                validate_features(df.copy())

            # Use C2 flows only
            if "dst_ip" in df.columns:
                c2=df[df["dst_ip"].astype(str).str.contains("104.16",na=False)]
                df_use=c2 if not c2.empty else df
            else:
                df_use=df

            m=map_features(df_use,fn)
            if len(m)>0: vecs.append(m[0])

            for f in [pcap,csv]:
                try: os.remove(f)
                except: pass

        if not vecs:
            print("  [!] No features extracted"); continue

        X=np.array(vecs)
        if scaler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X=scaler.transform(X)

        for mn,model in models.items():
            probs=model.predict_proba(X)[:,1]
            det=int((probs>0.5).sum())
            ev=1-det/len(probs)
            print(f"   {mn:6s} → evasion={ev:.1%}  conf={probs.mean():.3f}  n={len(probs)}")
            results.append({"strategy":strategy.value,"model":mn,
                           "evasion_rate":ev,
                           "mean_confidence":float(probs.mean()),
                           "n_flows":len(probs)})

    df_out=pd.DataFrame(results)
    df_out.to_csv("real_adversarial_results.csv",index=False)
    df_out.to_json("real_adversarial_results.json",orient="records",indent=2)
    print(f"\n[+] Saved to real_adversarial_results.csv")

    if not df_out.empty:
        print(f"\n{'='*65}")
        print("  RESULTS SUMMARY")
        print(f"{'='*65}")
        pivot=df_out.pivot_table(index="strategy",columns="model",
                                  values="evasion_rate")
        order=[s.value for s in EvasionStrategy]
        pivot=pivot.reindex([s for s in order if s in pivot.index])
        print(pivot.map(lambda x: f"{x:.1%}").to_string())

if __name__=="__main__":
    main()
