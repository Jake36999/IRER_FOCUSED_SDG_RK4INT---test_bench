import sqlite3
import shutil
import os

# --- PATH CONFIGURATION ---
F_DRIVE_ROOT = r"F:\GPU_PY_IRER_Hunter"
C_DRIVE_ROOT = r"C:\Users\jakem\Documents\GPU_PY_IRER_Hunter"

DB_PATH = os.path.join(F_DRIVE_ROOT, "simulation_ledger.db")

def sync_elite_artifacts():
    print(f"Connecting to Active Ledger on F: Drive...")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    # Connect to the F: drive database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # STEP 1: Query for all runs with at least one perfect harmonic match
    print("Querying for Elite Runs (SSE <= 6.0001)...")
    cursor.execute("""
        SELECT config_hash, log_prime_sse 
        FROM metrics 
        WHERE log_prime_sse <= 6.0001
    """)
    elites = cursor.fetchall()
    conn.close()

    print(f"Found {len(elites)} Elite Runs. Initiating Mirror to C: Drive...")

    # Ensure C: drive target directories exist
    os.makedirs(os.path.join(C_DRIVE_ROOT, "input_configs"), exist_ok=True)
    os.makedirs(os.path.join(C_DRIVE_ROOT, "simulation_data"), exist_ok=True)
    os.makedirs(os.path.join(C_DRIVE_ROOT, "provenance_reports"), exist_ok=True)

    success_count = 0
    # STEP 2: Copy the Artifacts
    for config_hash, sse in elites:
        files_to_copy = [
            (rf"input_configs\config_{config_hash}.json", rf"input_configs\config_{config_hash}.json"),
            (rf"simulation_data\rho_history_{config_hash}.h5", rf"simulation_data\rho_history_{config_hash}.h5"),
            (rf"provenance_reports\provenance_{config_hash}.json", rf"provenance_reports\provenance_{config_hash}.json")
        ]

        copied_all = True
        for src_rel, dest_rel in files_to_copy:
            src = os.path.join(F_DRIVE_ROOT, src_rel)
            dest = os.path.join(C_DRIVE_ROOT, dest_rel)
            
            if os.path.exists(src):
                # Only copy if it doesn't already exist on C: to save time
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
            else:
                copied_all = False
        
        if copied_all:
            success_count += 1

    print(f"\nExtraction Complete!")
    print(f"Successfully mirrored {success_count} complete Elite artifacts to your C: Drive.")

if __name__ == "__main__":
    sync_elite_artifacts()