import numpy as np
import quantulemapper_real as qm  # type: ignore[import]

def run_monte_carlo_p_value(target_sse: float, grid_shape: tuple = (16, 16, 16), n_iterations: int = 1000):
    """
    Generates N random 3D noise fields, calculates their prime-log SSE, 
    and returns the p-value of achieving the target_sse by chance.
    """
    print(f"[Monte Carlo] Running {n_iterations} null-hypothesis iterations...")
    random_sses = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        # Generate random Gaussian noise field
        random_field = np.random.normal(loc=1.0, scale=0.1, size=grid_shape)
        try:
            sse_result = qm.prime_log_sse(random_field)
            random_sses[i] = float(sse_result.get("log_prime_sse", 999.0))
        except Exception:
            random_sses[i] = 999.0
    # Calculate p-value: what fraction of random runs beat our target SSE?
    better_runs = np.sum(random_sses <= target_sse)
    p_value = better_runs / n_iterations
    print(f"  -> P-Value of SSE {target_sse:.6f}: {p_value:.6f}")
    return p_value, float(np.mean(random_sses))