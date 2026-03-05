"""
orchestrator/hunter.py
The Adaptive Hunter (Genetic Algorithm).
"""
import random
import uuid
import hashlib
import json
import numpy as np

class GeneticHunter:
    def __init__(self, kel_client):
        self.kel = kel_client
        self.generation = 1
        
        # V14 Default Search Space (SDG Params)
        self.defaults = {
            "dt": 0.01,
            "epsilon": 0.1,
            "alpha": 0.1,    # Geometric Coupling (Critical)
            "sigma_k": 1.0,  # Density Sensitivity
            "c1": 0.1,
            "c3": 1.0,
            "splash_fraction": 0.2,
            "dx": 1.0
        }

    def generate_batch(self, batch_size=5):
        """Produces a batch of job manifests with config_hash for provenance."""
        elites = self.kel.get_elites(limit=5)
        batch = []
        for _ in range(batch_size):
            if not elites or random.random() < 0.2:
                # 20% Diversity Injection (Random Search)
                params = self.mutate(self.defaults.copy(), intensity=1.0)
            else:
                # 80% Evolution (Mutate from Elite)
                parent = random.choice(elites)
                params = self.mutate(parent['params'], intensity=0.1)

            job_id = self.mint_job_id(params)
            config_hash = self.mint_config_hash(params)

            manifest = {
                "job_id": job_id,
                "generation": self.generation,
                "params": params,
                "grid_size": 32,      # Fixed for now, scale up later
                "total_steps": 1000,  # JAX Scan length
                "config_hash": config_hash
            }
            batch.append(manifest)
        self.generation += 1
        return batch

    def mint_config_hash(self, params):
        """Hash the params to create a config hash (full SHA256)."""
        s = json.dumps(params, sort_keys=True)
        import hashlib
        return hashlib.sha256(s.encode()).hexdigest()

    def mutate(self, params, intensity=0.1):
        """Applies Gaussian mutation to float parameters."""
        new_p = params.copy()
        keys_to_mutate = ['alpha', 'sigma_k', 'c1', 'c3', 'splash_fraction']
        
        for k in keys_to_mutate:
            val = float(new_p.get(k, 0.0))
            noise = np.random.normal(0, intensity * abs(val + 0.01))
            new_p[k] = abs(val + noise) # Parameters must be positive
            
        return new_p

    def mint_job_id(self, params):
        """Identity-as-Code: Hash the params to create the ID."""
        s = json.dumps(params, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]