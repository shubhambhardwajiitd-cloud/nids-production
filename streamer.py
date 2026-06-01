"""
streamer.py — Simulates real-time network traffic by streaming
CSV rows to the NIDS API one by one with a configurable delay.

Usage:
    py -3.11 streamer.py                         # uses defaults
    py -3.11 streamer.py --delay 0.5             # faster streaming
    py -3.11 streamer.py --model random_forest   # switch model
    py -3.11 streamer.py --limit 50              # limit rows
"""

import os
import argparse
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("NIDS_API_URL", "http://localhost:8000")
API_KEY   = "dev-key-free-001"
DATA_PATH = r"C:\Users\Shubham Bhardwaj\Desktop\Projects_tech\NIDS\data\sample_data.csv"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key":    API_KEY,
}

# ── Terminal colors ───────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "CRITICAL": "\033[91m",
    "HIGH":     "\033[93m",
    "MEDIUM":   "\033[94m",
    "LOW":      "\033[92m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"

def color_severity(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "")
    return f"{BOLD}{color}{severity}{RESET}"


def stream_csv(filepath: str, model_name: str, delay: float, limit: int):

    print(f"\n{'='*60}")
    print(f"  NIDS Real-Time Streamer")
    print(f"{'='*60}")
    print(f"  File:   {filepath}")
    print(f"  Model:  {model_name}")
    print(f"  Delay:  {delay}s between rows")
    print(f"  Limit:  {'unlimited' if limit == 0 else limit} rows")
    print(f"  API:    {API_URL}")
    print(f"{'='*60}\n")

    # ── Check API health ───────────────────────────────────────────
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ API is not healthy. Is uvicorn running?")
            return
        print(f"✅ API healthy — models: {health.json()['models_loaded']}\n")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Run uvicorn first:")
        print("   py -3.11 -m uvicorn main:app --reload --port 8000")
        return

    # ── Load CSV ───────────────────────────────────────────────────
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return

    true_labels = None
    if "Label" in df.columns:
        true_labels = df["Label"].values
        df = df.drop(columns=["Label"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    total_rows = len(df) if limit == 0 else min(limit, len(df))

    # ── Per-class tracking ─────────────────────────────────────────
    class_total   = defaultdict(int)
    class_correct = defaultdict(int)
    class_wrong   = defaultdict(lambda: defaultdict(int))  # true → predicted wrong

    attacks_found = 0
    total_sent    = 0

    print(f"📡 Streaming {total_rows} rows...\n")

    for i in range(total_rows):
        row        = df.iloc[i].to_dict()
        true_label = true_labels[i] if true_labels is not None else "unknown"

        try:
            response = requests.post(
                f"{API_URL}/predict?model_name={model_name}",
                headers = HEADERS,
                json    = {"features": row},
                timeout = 10,
            )
            total_sent += 1

            if response.status_code == 200:
                result     = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                severity   = result["severity"]
                is_attack  = result["is_attack"]
                shap       = result.get("shap_explanation", [])

                # ── Per-class tracking ─────────────────────────────
                if true_labels is not None:
                    class_total[true_label] += 1
                    if prediction == true_label:
                        class_correct[true_label] += 1
                    else:
                        class_wrong[true_label][prediction] += 1

                timestamp = datetime.now().strftime("%H:%M:%S")

                if is_attack:
                    attacks_found += 1
                    shap_str = " | ".join(
                        [f"{s['feature']}({s['shap_value']:+.2f})" for s in shap[:3]]
                    )
                    correct_str = "✓" if prediction == true_label else "✗"
                    print(
                        f"[{timestamp}] Row {i+1:>4} "
                        f"🚨 {color_severity(severity)} "
                        f"→ {BOLD}{prediction}{RESET} "
                        f"(conf: {confidence:.2f}) "
                        f"| True: {true_label} {correct_str}\n"
                        f"           SHAP: {shap_str}\n"
                    )
                else:
                    if i % 10 == 0:
                        correct_str = "✓" if true_label == "BENIGN" else "✗"
                        print(
                            f"[{timestamp}] Row {i+1:>4} "
                            f"✅ BENIGN "
                            f"(conf: {confidence:.2f}) "
                            f"| True: {true_label} {correct_str}"
                        )
            else:
                print(f"[Row {i+1}] ❌ API error {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"[Row {i+1}] ⏱ Timed out — skipping")
        except requests.exceptions.ConnectionError:
            print(f"[Row {i+1}] ❌ Lost connection — stopping")
            break

        time.sleep(delay)

    # ── Per-class detection report ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Streaming Complete — Per-Class Detection Report")
    print(f"{'='*60}")
    print(f"  {'Class':<25} {'Detected':<10} {'Rate':<8} {'Confused With'}")
    print(f"  {'-'*55}")

    for label in sorted(class_total.keys()):
        total   = class_total[label]
        correct = class_correct[label]
        rate    = correct / total * 100 if total > 0 else 0
        wrong   = class_wrong[label]

        # Show what it was confused with
        confused = ""
        if wrong:
            top_confusion = max(wrong, key=wrong.get)
            confused = f"→ {top_confusion} ({wrong[top_confusion]}x)"

        status = "✅" if rate >= 70 else "⚠️" if rate >= 40 else "❌"
        print(f"  {status} {label:<23} {correct}/{total:<8} {rate:>5.0f}%   {confused}")

    print(f"\n  Rows sent:      {total_sent}")
    print(f"  Attacks found:  {attacks_found}")
    print(f"  Benign flows:   {total_sent - attacks_found}")
    print(f"{'='*60}\n")

    # ── API stats ──────────────────────────────────────────────────
    try:
        stats = requests.get(
            f"{API_URL}/stats",
            headers={"x-api-key": API_KEY},
            timeout=5
        ).json()
        print("📊 API Alert Store:")
        print(f"   Total alerts: {stats['total_alerts']}")
        for attack, count in sorted(
            stats["attack_distribution"].items(),
            key=lambda x: x[1], reverse=True
        ):
            print(f"   {attack:<25} {count} alerts")
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIDS Real-Time Streamer")
    parser.add_argument("--file",  type=str,   default=DATA_PATH)
    parser.add_argument("--model", type=str,   default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int,   default=0)

    args = parser.parse_args()
    stream_csv(
        filepath   = args.file,
        model_name = args.model,
        delay      = args.delay,
        limit      = args.limit,
    )
