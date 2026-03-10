#!/usr/bin/env python3
import json
import os
import pandas as pd

# --- DIRECTORY BINDINGS ---
INPUT_CSV = "long run results (lw_SSE_1.93).csv"
CONFIGS_DIR = r"F:\GPU_PY_IRER_Hunter\input_configs"
OUTPUT_QUEUE = "backlog_queue.json"
SSE_THRESHOLD = 7.0

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ File {INPUT_CSV} not found. Please ensure it is in the same directory.")
        return
    
    if not os.path.exists(CONFIGS_DIR):
        print(f"❌ Configs directory {CONFIGS_DIR} not found. Ensure the F: Drive is connected.")
        return

    # 1. Read the exported SQLite CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return
        
    # 2. Filter for SSE <= 7.0 and sort best-first
    elite_df = df[df['log_prime_sse'] <= SSE_THRESHOLD].sort_values(by='log_prime_sse')
    
    clean_queue = []
    missing_configs = 0

    # 3. Map the Hash to the Config JSONs to extract the physics parameters
    print(f"🔍 Scanning {len(elite_df)} elite candidates...")
    for _, row in elite_df.iterrows():
        c_hash = str(row['config_hash']).strip()
        config_path = os.path.join(CONFIGS_DIR, f"config_{c_hash}.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = json.load(f)
            
            # Extract only the physics hyperparameters
            params = {"origin": "BACKLOG_RE_SIM"}
            for k, v in full_config.items():
                if k.startswith("param_"):
                    params[k] = float(v)
            
            clean_queue.append(params)
        else:
            missing_configs += 1

    # 4. Save the finalized queue for the Orchestrator
    with open(OUTPUT_QUEUE, 'w') as f:
        json.dump(clean_queue, f, indent=4)

    print(f"✅ Successfully queued {len(clean_queue)} elite configurations into {OUTPUT_QUEUE}")
    if missing_configs > 0:
        print(f"⚠️ Warning: {missing_configs} config JSONs were missing from the F: Drive input_configs directory.")

if __name__ == "__main__":
    main()