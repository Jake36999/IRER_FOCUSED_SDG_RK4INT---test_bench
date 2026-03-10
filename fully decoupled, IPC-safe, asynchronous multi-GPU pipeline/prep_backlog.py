#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Any

# --- DIRECTORY BINDINGS ---
DEFAULT_INPUT_CSV = r"E:\Development_back_up_folder_2026\long_run data back up\merged_parameters_results.csv"
DEFAULT_OUTPUT = "backlog_queue.json"
DEFAULT_SSE_THRESHOLD = 15.0

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare backlog_queue.json from merged CSV.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Path to merged results CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output backlog queue JSON path.")
    parser.add_argument("--sse-threshold", type=float, default=DEFAULT_SSE_THRESHOLD, help="Keep rows where log_prime_sse <= threshold.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on number of queued configs.")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"❌ File {args.input_csv} not found.")
        return

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return
        
    elite_df = df[df['log_prime_sse'] <= args.sse_threshold].sort_values(by='log_prime_sse')
    
    clean_queue = []

    print(f"🔍 Scanning {len(elite_df)} elite candidates...")
    for _, row in elite_df.iterrows():
        c_hash = str(row.get('config_hash', 'UNKNOWN')).strip()
        
        # --- INJECT GEOMETRY DIRECTLY INTO THE PAYLOAD ---
        params: Dict[str, Any] = {
            "origin": "BACKLOG_RE_SIM",
            "config_hash": c_hash,
            "simulation": {
                "N_grid": 128,
                "L_domain": 10.0,
                "T_steps": 1200,
                "dt": 0.0005,
                "collapse_threshold": 10000000000.0
            }
        }
        
        # Extract numeric params
        for k in row.index:
            if k.startswith("param_"):
                try:
                    params[k] = float(row[k])
                except:
                    pass
        
        clean_queue.append(params)

    if args.max_items is not None:
        clean_queue = clean_queue[:args.max_items]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(clean_queue, f, indent=4)

    print(f"✅ Successfully queued {len(clean_queue)} elite configurations into {args.output}")

if __name__ == "__main__":
    main()