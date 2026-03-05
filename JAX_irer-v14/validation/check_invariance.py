"""
worker/validation/check_invariance.py
Verifies that the physics scales correctly with resolution (Grid Invariance).
"""
import jax
import jax.numpy as jnp
from worker.ir_physics import solver, models, kernels

def run_invariance_test():
    print("Running Resolution Invariance Test...")
    
    # Shared Physical Constants
    L_physical = 32.0 # Physical size of the box
    TOTAL_TIME = 20.0 # Physical duration
    
    # --- Run 1: Low Resolution (N=32) ---
    N_low = 32
    dx_low = L_physical / N_low
    # Scaling dt with dx^2 for stability
    dt_low = 0.05 * (dx_low**2) 
    steps_low = int(TOTAL_TIME / dt_low)
    
    print(f"\n--- Low Res Run (N={N_low}, dx={dx_low:.3f}, dt={dt_low:.4f}) ---")
    
    params_low = models.SimParams(
        dt=dt_low, epsilon=0.1, alpha=0.5, rho_vac=1.0,
        c1=0.0, c3=1.0, splash_fraction=0.1, sigma_k=1.0, dx=dx_low
    )
    
    # Init state (Gaussian pulse)
    x = jnp.linspace(0, L_physical, N_low, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    center = L_physical / 2.0
    field_low = jnp.exp(-((X-center)**2 + (Y-center)**2 + (Z-center)**2)/16.0) + 0j
    
    state_low = models.SimState(
        time_idx=0, field=field_low, 
        omega=jnp.ones_like(field_low), 
        grad_omega=jnp.zeros((*field_low.shape, 3)), 
        h_norm=jnp.array(0.0), config_hash=0
    )
    
    final_low, _ = solver.run_simulation_scan(state_low, params_low, steps_low, N_low, use_adaptive_dt=True)
    
    # --- Run 2: High Resolution (N=64) ---
    N_high = 64
    dx_high = L_physical / N_high
    # dt must scale with dx^2 to maintain same CFL condition
    dt_high = 0.05 * (dx_high**2)
    steps_high = int(TOTAL_TIME / dt_high)
    
    print(f"\n--- High Res Run (N={N_high}, dx={dx_high:.3f}, dt={dt_high:.4f}) ---")
    
    params_high = models.SimParams(
        dt=dt_high, epsilon=0.1, alpha=0.5, rho_vac=1.0,
        c1=0.0, c3=1.0, splash_fraction=0.1, sigma_k=1.0, dx=dx_high
    )
    
    # Init state (Same physical Gaussian, higher sampling)
    x = jnp.linspace(0, L_physical, N_high, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    field_high = jnp.exp(-((X-center)**2 + (Y-center)**2 + (Z-center)**2)/16.0) + 0j
    
    state_high = models.SimState(
        time_idx=0, field=field_high, 
        omega=jnp.ones_like(field_high), 
        grad_omega=jnp.zeros((*field_high.shape, 3)), 
        h_norm=jnp.array(0.0), config_hash=0
    )
    
    final_high, _ = solver.run_simulation_scan(state_high, params_high, steps_high, N_high, use_adaptive_dt=True)

    # --- Comparison ---
    print("\n--- Invariance metrics ---")
    max_rho_low = jnp.max(jnp.abs(final_low.field)**2)
    max_rho_high = jnp.max(jnp.abs(final_high.field)**2)
    
    print(f"Max Density (Low Res):  {max_rho_low:.4f}")
    print(f"Max Density (High Res): {max_rho_high:.4f}")
    
    diff = jnp.abs(max_rho_low - max_rho_high)
    print(f"Absolute Difference: {diff:.4f}")
    
    if diff < 0.5: # Tolerance depends on dynamics, but shouldn't be massive
        print("✅ Grid Invariance Confirmed (Physics scales correctly)")
    else:
        print("⚠️ Significant Deviation (Check scaling laws)")

if __name__ == "__main__":
    run_invariance_test()