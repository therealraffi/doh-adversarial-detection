"""
real_adversarial_pipeline.py
"""
import os, sys, time, random, subprocess, json, argparse
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_shaper import TrafficShaper, EvasionStrategy

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

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

def schedule_to_packets(schedule, start_time):
    src_mac="00:1A:2B:3C:4D:5E"; src_ip="192.168.1.50"
    src_port=random.randint(1024,65535); dst_port=443
    packets=[]; t=start_time
    for q in schedule:
        t += q.delay_ms/1000.0
        dst_ip="8.8.8.8" if q.is_cover else "104.16.248.249"
        dst_mac="AA:BB:CC:DD:EE:00" if q.is_cover else "AA:BB:CC:DD:EE:11"
        req_size=max(40, q.payload_bytes+q.padding_bytes+40)
        req=Ether(src=src_mac,dst=dst_mac)/IP(src=src_ip,dst=dst_ip)/TCP(sport=src_port,dport=dst_port,flags='PA',seq=1001)/Raw(b"X"*req_size)
        req.time=t; packets.append(req)
        t+=random.uniform(0.02,0.15)
        resp_size=max(100,int(req_size*random.uniform(1.5,4.0)))
        resp=Ether(src=dst_mac,dst=src_mac)/IP(src=dst_ip,dst=src_ip)/TCP(sport=dst_port,dport=src_port,flags='PA',seq=2001)/Raw(b"Y"*resp_size)
        resp.time=t; packets.append(resp)
    return packets

def run_cicflowmeter(pcap_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        subprocess.run(["cicflowmeter","-f",pcap_path,"-c",output_dir+".csv"], capture_output=True, timeout=60)
    except:
        try:
            subprocess.run(["python","-m","cicflowmeter","-f",pcap_path,"-c",output_dir+".csv"], capture_output=True, timeout=60)
        except Exception as e:
            print(f"[!] cicflowmeter error: {e}"); return pd.DataFrame()
    csvs=[output_dir+".csv"] if os.path.exists(output_dir+".csv") else []
    if not csvs or not os.path.exists(csvs[0]): return pd.DataFrame()
    df=pd.read_csv(csvs[0])
    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    return df

def map_features(df, feature_names):
    rows=[]
    for _,row in df.iterrows():
        pm=row.get("pkt_len_mean",row.get("average_packet_size",0))
        ps=row.get("pkt_len_std",row.get("packet_length_std",0))
        pv=row.get("pkt_len_var",ps**2)
        px=row.get("pkt_len_max",pm); pn=row.get("pkt_len_min",pm)
        im=row.get("flow_iat_mean",0); is_=row.get("flow_iat_std",0)
        iv=row.get("flow_iat_var",is_**2)
        ix=row.get("flow_iat_max",im); ino=row.get("flow_iat_min",im)
        rm=row.get("fwd_iat_mean",im); rs=row.get("fwd_iat_std",is_)
        rv=row.get("fwd_iat_var",rs**2)
        rx=row.get("fwd_iat_max",rm); rn=row.get("fwd_iat_min",rm)
        feat={
            "FlowBytesSent":row.get("totlen_fwd_pkts",0),
            "FlowBytesReceived":row.get("totlen_bwd_pkts",0),
            "FlowSentRate":row.get("flow_byts_s",0),
            "FlowReceivedRate":row.get("flow_pkts_s",0),
            "PacketLengthMean":pm,"PacketLengthVariance":pv,
            "PacketLengthStandardDeviation":ps,
            "PacketLengthMedian":(px+pn)/2,"PacketLengthMode":pm,
            "PacketLengthSkewFromMedian":(pm-(px+pn)/2)/(ps+1e-9),
            "PacketLengthSkewFromMode":0.0,
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",required=True)
    parser.add_argument("--flows",type=int,default=20)
    args=parser.parse_args()

    fn=json.load(open(os.path.join(args.results,"feature_names.json"))) if os.path.exists(os.path.join(args.results,"feature_names.json")) else DETECTOR_FEATURES
    scaler=joblib.load(os.path.join(args.results,"scaler.joblib")) if os.path.exists(os.path.join(args.results,"scaler.joblib")) else None
    models={}
    for n,f in [("RF","rf.joblib"),("GB","gb.joblib"),("XGB","xgb.joblib")]:
        p=os.path.join(args.results,f)
        if os.path.exists(p): models[n]=joblib.load(p); print(f"[+] {n} loaded")

    shaper=TrafficShaper()
    payload=b"RESEARCH:"+b"X"*503
    results=[]

    print(f"\n{'='*65}")
    print("  REAL ADVERSARIAL PIPELINE — Scapy+CICFlowMeter")
    print(f"{'='*65}")

    for strategy in EvasionStrategy:
        print(f"\n── {strategy.value} ──────────────────────")
        vecs=[]
        for i in range(args.flows):
            sched=shaper.build_schedule(payload,strategy=strategy,chunk_size=40)
            pkts=schedule_to_packets(sched, time.time()+i*2.0)
            if not pkts: continue
            pcap=f"/tmp/adv_{strategy.value}_{i}.pcap"
            wrpcap(pcap,pkts)
            df=run_cicflowmeter(pcap,f"/tmp/cic_{strategy.value}_{i}")
            if df.empty: continue
            m=map_features(df,fn)
            if len(m)>0: vecs.append(m[0])
            try: os.remove(pcap)
            except: pass

        if not vecs:
            print("  [!] No features extracted"); continue

        X=np.array(vecs)
        if scaler: X=scaler.transform(X)

        for mn,model in models.items():
            probs=model.predict_proba(X)[:,1]
            det=int((probs>0.5).sum())
            ev=1-det/len(probs)
            print(f"   {mn:6s} → evasion={ev:.1%}  conf={probs.mean():.3f}")
            results.append({"strategy":strategy.value,"model":mn,"evasion_rate":ev,"mean_confidence":float(probs.mean()),"n_flows":len(probs)})

    pd.DataFrame(results).to_csv("real_adversarial_results.csv",index=False)
    print("\n[+] Saved to real_adversarial_results.csv")

if __name__=="__main__":
    main()
