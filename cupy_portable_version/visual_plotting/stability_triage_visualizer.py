import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def generate_triage_heatmaps(db_path="simulation_ledger.db"):
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    # SQL Query: Join parameters and metrics for completed runs
    query = """
    SELECT 
        p.param_D, p.param_eta, p.param_a_coupling, 
        m.log_prime_sse, m.pcs, m.ic, r.fitness
    FROM runs r
    JOIN parameters p ON r.config_hash = p.config_hash
    JOIN metrics m ON r.config_hash = m.config_hash
    WHERE r.status = 'completed' AND m.log_prime_sse < 900.0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("Not enough valid data in the database to generate heatmaps.")
        return

    # Create Inverse IC metric
    df['inverse_ic'] = 1.0 / df['ic'].clip(lower=1e-12)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: The Spectral Attractor Basin (SSE)
    sc1 = axes[0].scatter(df['param_D'], df['param_eta'], c=df['log_prime_sse'], cmap='viridis_r', s=100, edgecolor='black')
    axes[0].set_title('Spectral Attractor Basin (SSE)')
    axes[0].set_xlabel('Kinetic Diffusion (param_D)')
    axes[0].set_ylabel('Friction / Damping (param_eta)')
    fig.colorbar(sc1, ax=axes[0], label='Log-Prime SSE (Lower is Better)')

    # Plot 2: Phase Coherence Map (PCS)
    sc2 = axes[1].scatter(df['param_D'], df['param_eta'], c=df['pcs'], cmap='plasma', s=100, edgecolor='black')
    axes[1].set_title('Topological Stability (Phase Coherence)')
    axes[1].set_xlabel('Kinetic Diffusion (param_D)')
    fig.colorbar(sc2, ax=axes[1], label='PCS (Target = 1.0)')

    # Plot 3: Inverse Informational Current (Confinement)
    # Using log scale for IC^-1 to highlight extreme confinement
    sc3 = axes[2].scatter(df['param_D'], df['param_eta'], c=np.log10(df['inverse_ic']), cmap='magma', s=100, edgecolor='black')
    axes[2].set_title('Confinement Zone (Log Inverse IC)')
    axes[2].set_xlabel('Kinetic Diffusion (param_D)')
    fig.colorbar(sc3, ax=axes[2], label='Log10(IC^-1) (Higher = Tighter Confinement)')

    plt.tight_layout()
    plt.savefig('ASTE_V11_Triage_Heatmaps.png', dpi=300)
    print("✅ Successfully generated 'ASTE_V11_Triage_Heatmaps.png'")
    plt.show()

if __name__ == "__main__":
    generate_triage_heatmaps()
