"""
pcap_to_features.py — Extracts CIC-IDS-2018 compatible flow features
from a PCAP file and sends them to the NIDS API for prediction.

Usage:
    py -3.11 pcap_to_features.py --pcap capture.pcap
    py -3.11 pcap_to_features.py --pcap capture.pcap --send   (sends to API)
    py -3.11 pcap_to_features.py --pcap capture.pcap --out flows.csv
"""

import argparse
import math
import time
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
API_KEY = "dev-key-free-001"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key":    API_KEY,
}

# ── Flow timeout (seconds) ────────────────────────────────────────────────────
FLOW_TIMEOUT = 120

# ── Feature names matching CIC-IDS-2018 ──────────────────────────────────────
FEATURE_NAMES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


def safe_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def extract_flows(pcap_path):
    """Extract flow features from pcap file using scapy."""
    try:
        from scapy.all import rdpcap, IP, IPv6, TCP, UDP
    except ImportError:
        print("Installing scapy...")
        import subprocess
        subprocess.run(["pip", "install", "scapy", "--prefer-binary", "-q"])
        from scapy.all import rdpcap, IP, IPv6, TCP, UDP

    print(f"Reading {pcap_path}...")
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading pcap: {e}")
        return []

    print(f"Total packets: {len(packets)}")

    # ── Group packets into flows ───────────────────────────────────────────────
    flows = defaultdict(list)

    for pkt in packets:
        try:
            # Handle both IPv4 and IPv6
            if IP in pkt:
                src_ip   = pkt[IP].src
                dst_ip   = pkt[IP].dst
                protocol = pkt[IP].proto
            elif IPv6 in pkt:
                src_ip   = pkt[IPv6].src
                dst_ip   = pkt[IPv6].dst
                protocol = pkt[IPv6].nh
            else:
                continue

            src_port = 0
            dst_port = 0
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            # Flow key — bidirectional
            fwd_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            bwd_key = (dst_ip, src_ip, dst_port, src_port, protocol)

            # Determine direction
            if fwd_key in flows or bwd_key not in flows:
                key       = fwd_key
                direction = "fwd"
            else:
                key       = bwd_key
                direction = "bwd"

            flows[key].append({
                "time":      float(pkt.time),
                "direction": direction,
                "length":    len(pkt),
                "pkt":       pkt,
                "src_port":  src_port,
                "dst_port":  dst_port,
                "protocol":  protocol,
            })
        except Exception:
            continue

    print(f"Flows found: {len(flows)}")

    # ── Extract features per flow ──────────────────────────────────────────────
    feature_rows = []

    for flow_key, pkts in flows.items():
        if len(pkts) < 2:
            continue

        pkts = sorted(pkts, key=lambda x: x["time"])

        fwd_pkts = [p for p in pkts if p["direction"] == "fwd"]
        bwd_pkts = [p for p in pkts if p["direction"] == "bwd"]

        if not fwd_pkts:
            continue

        # Times
        all_times  = [p["time"] for p in pkts]
        fwd_times  = [p["time"] for p in fwd_pkts]
        bwd_times  = [p["time"] for p in bwd_pkts]

        flow_start    = all_times[0]
        flow_end      = all_times[-1]
        flow_duration = max((flow_end - flow_start) * 1e6, 1)  # microseconds

        # Packet lengths
        fwd_lens = [p["length"] for p in fwd_pkts]
        bwd_lens = [p["length"] for p in bwd_pkts]
        all_lens = fwd_lens + bwd_lens

        # IAT (inter-arrival times)
        all_iats = [all_times[i+1] - all_times[i]
                    for i in range(len(all_times)-1)]
        fwd_iats = [fwd_times[i+1] - fwd_times[i]
                    for i in range(len(fwd_times)-1)]
        bwd_iats = [bwd_times[i+1] - bwd_times[i]
                    for i in range(len(bwd_times)-1)]

        # Flags
        fin_cnt = psh_cnt = syn_cnt = rst_cnt = ack_cnt = urg_cnt = 0
        fwd_psh = bwd_psh = fwd_urg = bwd_urg = 0
        fwd_hdr_len = bwd_hdr_len = 0
        init_fwd_win = init_bwd_win = -1
        fwd_act_data = 0
        fwd_seg_min  = 0

        for p in pkts:
            pkt = p["pkt"]
            if TCP in pkt:
                flags = pkt[TCP].flags
                if flags & 0x01: fin_cnt += 1
                if flags & 0x02: syn_cnt += 1
                if flags & 0x04: rst_cnt += 1
                if flags & 0x08:
                    psh_cnt += 1
                    if p["direction"] == "fwd": fwd_psh += 1
                    else: bwd_psh += 1
                if flags & 0x10: ack_cnt += 1
                if flags & 0x20:
                    urg_cnt += 1
                    if p["direction"] == "fwd": fwd_urg += 1
                    else: bwd_urg += 1

                hdr = pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
                if p["direction"] == "fwd":
                    fwd_hdr_len += hdr
                    if init_fwd_win == -1:
                        init_fwd_win = pkt[TCP].window
                    payload = len(pkt[TCP].payload)
                    if payload > 0:
                        fwd_act_data += 1
                    fwd_seg_min = min(fwd_seg_min or hdr, hdr)
                else:
                    bwd_hdr_len += hdr
                    if init_bwd_win == -1:
                        init_bwd_win = pkt[TCP].window

        # Rates
        dur_sec      = flow_duration / 1e6
        flow_byts_s  = sum(all_lens) / dur_sec if dur_sec > 0 else 0
        flow_pkts_s  = len(pkts) / dur_sec if dur_sec > 0 else 0
        fwd_pkts_s   = len(fwd_pkts) / dur_sec if dur_sec > 0 else 0
        bwd_pkts_s   = len(bwd_pkts) / dur_sec if dur_sec > 0 else 0

        down_up = len(bwd_pkts) / len(fwd_pkts) if fwd_pkts else 0

        row = {
            "Dst Port":         flow_key[3],
            "Protocol":         flow_key[4],
            "Flow Duration":    flow_duration,
            "Tot Fwd Pkts":     len(fwd_pkts),
            "Tot Bwd Pkts":     len(bwd_pkts),
            "TotLen Fwd Pkts":  sum(fwd_lens),
            "TotLen Bwd Pkts":  sum(bwd_lens),
            "Fwd Pkt Len Max":  max(fwd_lens) if fwd_lens else 0,
            "Fwd Pkt Len Min":  min(fwd_lens) if fwd_lens else 0,
            "Fwd Pkt Len Mean": safe_mean(fwd_lens),
            "Fwd Pkt Len Std":  safe_std(fwd_lens),
            "Bwd Pkt Len Max":  max(bwd_lens) if bwd_lens else 0,
            "Bwd Pkt Len Min":  min(bwd_lens) if bwd_lens else 0,
            "Bwd Pkt Len Mean": safe_mean(bwd_lens),
            "Bwd Pkt Len Std":  safe_std(bwd_lens),
            "Flow Byts/s":      flow_byts_s,
            "Flow Pkts/s":      flow_pkts_s,
            "Flow IAT Mean":    safe_mean(all_iats) * 1e6 if all_iats else 0,
            "Flow IAT Std":     safe_std(all_iats) * 1e6 if all_iats else 0,
            "Flow IAT Max":     max(all_iats) * 1e6 if all_iats else 0,
            "Flow IAT Min":     min(all_iats) * 1e6 if all_iats else 0,
            "Fwd IAT Tot":      sum(fwd_iats) * 1e6 if fwd_iats else 0,
            "Fwd IAT Mean":     safe_mean(fwd_iats) * 1e6 if fwd_iats else 0,
            "Fwd IAT Std":      safe_std(fwd_iats) * 1e6 if fwd_iats else 0,
            "Fwd IAT Max":      max(fwd_iats) * 1e6 if fwd_iats else 0,
            "Fwd IAT Min":      min(fwd_iats) * 1e6 if fwd_iats else 0,
            "Bwd IAT Tot":      sum(bwd_iats) * 1e6 if bwd_iats else 0,
            "Bwd IAT Mean":     safe_mean(bwd_iats) * 1e6 if bwd_iats else 0,
            "Bwd IAT Std":      safe_std(bwd_iats) * 1e6 if bwd_iats else 0,
            "Bwd IAT Max":      max(bwd_iats) * 1e6 if bwd_iats else 0,
            "Bwd IAT Min":      min(bwd_iats) * 1e6 if bwd_iats else 0,
            "Fwd PSH Flags":    fwd_psh,
            "Bwd PSH Flags":    bwd_psh,
            "Fwd URG Flags":    fwd_urg,
            "Bwd URG Flags":    bwd_urg,
            "Fwd Header Len":   fwd_hdr_len,
            "Bwd Header Len":   bwd_hdr_len,
            "Fwd Pkts/s":       fwd_pkts_s,
            "Bwd Pkts/s":       bwd_pkts_s,
            "Pkt Len Min":      min(all_lens) if all_lens else 0,
            "Pkt Len Max":      max(all_lens) if all_lens else 0,
            "Pkt Len Mean":     safe_mean(all_lens),
            "Pkt Len Std":      safe_std(all_lens),
            "Pkt Len Var":      safe_std(all_lens) ** 2,
            "FIN Flag Cnt":     fin_cnt,
            "SYN Flag Cnt":     syn_cnt,
            "RST Flag Cnt":     rst_cnt,
            "PSH Flag Cnt":     psh_cnt,
            "ACK Flag Cnt":     ack_cnt,
            "URG Flag Cnt":     urg_cnt,
            "CWE Flag Count":   0,
            "ECE Flag Cnt":     0,
            "Down/Up Ratio":    down_up,
            "Pkt Size Avg":     safe_mean(all_lens),
            "Fwd Seg Size Avg": safe_mean(fwd_lens),
            "Bwd Seg Size Avg": safe_mean(bwd_lens),
            "Fwd Byts/b Avg":   0,
            "Fwd Pkts/b Avg":   0,
            "Fwd Blk Rate Avg": 0,
            "Bwd Byts/b Avg":   0,
            "Bwd Pkts/b Avg":   0,
            "Bwd Blk Rate Avg": 0,
            "Subflow Fwd Pkts": len(fwd_pkts),
            "Subflow Fwd Byts": sum(fwd_lens),
            "Subflow Bwd Pkts": len(bwd_pkts),
            "Subflow Bwd Byts": sum(bwd_lens),
            "Init Fwd Win Byts": init_fwd_win if init_fwd_win != -1 else 0,
            "Init Bwd Win Byts": init_bwd_win if init_bwd_win != -1 else 0,
            "Fwd Act Data Pkts": fwd_act_data,
            "Fwd Seg Size Min":  fwd_seg_min,
            "Active Mean":       0,
            "Active Std":        0,
            "Active Max":        0,
            "Active Min":        0,
            "Idle Mean":         0,
            "Idle Std":          0,
            "Idle Max":          0,
            "Idle Min":          0,
        }
        feature_rows.append(row)

    return feature_rows


def send_to_api(features, model_name="xgboost"):
    """Send a single flow to the NIDS API."""
    try:
        response = requests.post(
            f"{API_URL}/predict?model_name={model_name}",
            headers=HEADERS,
            json={"features": features},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"API error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="PCAP to NIDS Features")
    parser.add_argument("--pcap",  required=True, help="Path to pcap file")
    parser.add_argument("--out",   default="live_capture.csv", help="Output CSV path")
    parser.add_argument("--send",  action="store_true", help="Send to API")
    parser.add_argument("--model", default="xgboost", help="Model to use")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PCAP Feature Extractor")
    print(f"{'='*60}")
    print(f"  PCAP:  {args.pcap}")
    print(f"  Output: {args.out}")
    print(f"  Send to API: {args.send}")
    print(f"{'='*60}\n")

    # Extract flows
    flows = extract_flows(args.pcap)

    if not flows:
        print("❌ No flows extracted — check pcap file")
        return

    print(f"\n✅ Extracted {len(flows)} flows")

    # Save to CSV
    df = pd.DataFrame(flows)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    df.to_csv(args.out, index=False)
    print(f"✅ Saved to {args.out}")

    # Send to API
    if args.send:
        print(f"\n📡 Sending {len(flows)} flows to API...\n")
        attacks = 0
        for i, flow in enumerate(flows):
            result = send_to_api(flow, args.model)
            if result:
                pred = result["prediction"]
                conf = result["confidence"]
                sev  = result["severity"]
                if result["is_attack"]:
                    attacks += 1
                    shap = result.get("shap_explanation", [])
                    shap_str = " | ".join(
                        [f"{s['feature']}({s['shap_value']:+.2f})"
                         for s in shap[:3]]
                    )
                    print(f"[{i+1:>4}] 🚨 {sev} → {pred} (conf:{conf:.2f})")
                    print(f"       SHAP: {shap_str}\n")
                else:
                    if i % 20 == 0:
                        print(f"[{i+1:>4}] ✅ BENIGN (conf:{conf:.2f})")
            time.sleep(args.delay)

        print(f"\n{'='*60}")
        print(f"  Done — {attacks} attacks found in {len(flows)} flows")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
