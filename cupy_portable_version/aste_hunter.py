#!/usr/bin/env python3
from typing import List, Dict, Any, Optional
import os
import json
import sqlite3
import random
import uuid
import math
import pandas  # type: ignore[import]
import pandas as pd #ignore type: import error, used for DataFrame handling in Hunter

"""
aste_hunter.py (SQLite DB Edition)
CLASSIFICATION: Scientific Adaptive Intelligence Engine (ASTE V12.1)
GOAL: Full NSGA-II Implementation. Features explicit Fast Non-Dominated Sorting, 
      Crowding Distance for diversity preservation, and Rank+Crowding selection.
"""

PROVENANCE_DIR = "provenance_reports"

# --- Configuration ---
DB_FILENAME = "simulation_ledger.db"
SSE_METRIC_KEY = "log_prime_sse"
HASH_KEY = "config_hash"

# Evolutionary Algorithm Parameters
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05

# Falsifiability Scaling (Decoupled to prevent Quantum/Classical override)
LAMBDA_CLASSICAL = 10.0
LAMBDA_QUANTUM = 5.0

class Hunter:
    """
    Implements the core evolutionary 'hunt' logic using a relational SQLite database.
    Manages population parameters, fitness tracking, and multi-objective lineage.
    """

    def export_ledger_to_json(self, output_path: str):
        """[DATA CONTRACT] Dumps the entire SQLite ledger to a formatted JSON file."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM results")
                rows = cursor.fetchall()
                ledger_data = [dict(row) for row in rows]
                with open(output_path, 'w') as f:
                    json.dump(ledger_data, f, indent=4)
        except Exception as e:
            print(f"[Hunter Error] Failed to export JSON ledger: {e}")

    def __init__(self, db_file: str = DB_FILENAME):
        self.db_file = db_file
        self._init_db()

    def _get_connection(self):
        """Enforces WAL mode on every single connection for CPU multi-threading safety."""
        conn = sqlite3.connect(self.db_file, timeout=15.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Creates the relational schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Core Run Tracking & Lineage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    status TEXT DEFAULT 'pending',
                    fitness REAL DEFAULT 0.0,
                    parent_1 TEXT,
                    parent_2 TEXT
                )
            ''')
            # Table 2: Physical Parameters
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    config_hash TEXT PRIMARY KEY,
                    param_D REAL,
                    param_eta REAL,
                    param_rho_vac REAL,
                    param_a_coupling REAL,
                    param_splash_coupling REAL,
                    param_splash_fraction REAL,
                    FOREIGN KEY(config_hash) REFERENCES runs(config_hash)
                )
            ''')
            
            # Table 3: Extracted Scientific Metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    config_hash TEXT PRIMARY KEY,
                    log_prime_sse REAL,
                    sse_null_phase_scramble REAL,
                    sse_null_target_shuffle REAL,
                    pcs REAL,
                    pli REAL,
                    ic REAL,
                    c4_contrast REAL,
                    ablated_c4_contrast REAL,
                    j_info_mean REAL,
                    grad_phase_var REAL,
                    max_amp_peak REAL,
                    clamp_fraction_mean REAL,
                    omega_sat_mean REAL,
                    FOREIGN KEY(config_hash) REFERENCES runs(config_hash)
                )
            ''')
            # Table 4: Results Ledger (for UI/telemetry)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    param_D REAL,
                    param_eta REAL,
                    param_rho_vac REAL,
                    param_a_coupling REAL,
                    param_splash_coupling REAL,
                    param_splash_fraction REAL,
                    log_prime_sse REAL,
                    fitness REAL,
                    parent_1 TEXT,
                    parent_2 TEXT
                )
            ''')
            # Table 5: Pareto Archive (Protects ALL historical non-dominated frontiers)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pareto_archive (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    log_prime_sse REAL,
                    fitness REAL
                )
            ''')
            
            # HPC Optimization: Indexing
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generation ON runs(generation);")

            # --- LIVE MIGRATION LOGIC FOR EXISTING DATABASES ---
            try:
                cursor.execute("ALTER TABLE metrics ADD COLUMN primary_harmonic_error REAL DEFAULT 999.0")
                cursor.execute("ALTER TABLE metrics ADD COLUMN missing_peak_penalty REAL DEFAULT 0.0")
                cursor.execute("ALTER TABLE metrics ADD COLUMN noise_penalty REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass # Columns already exist
                
            try:
                # NEW: Track LOM events
                cursor.execute("ALTER TABLE metrics ADD COLUMN collapse_event_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass 

            # --- LIVE MIGRATION LOGIC FOR SYNTHETIC LINEAGE ---
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'NATURAL'")
            except sqlite3.OperationalError:
                pass
            
            conn.commit()
            
    def get_current_generation(self) -> int:
        """Finds the highest generation number in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(generation) FROM runs WHERE status='completed'")
            result = cursor.fetchone()[0]
            return result if result is not None else -1

    def resume_hunt_state(self) -> int:
        return self.get_current_generation()

    def generate_random_parameters(self, bounds: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
        """Generates random Gen 0 parameters within specified bounds."""
        default_bounds = {
            "param_D": (0.01, 2.0),
            "param_eta": (0.01, 1.0),
            "param_rho_vac": (0.5, 1.5),
            "param_a_coupling": (0.1, 1.0),
            "param_splash_coupling": (0.0, 1.0),
            "param_splash_fraction": (-1.0, 0.0)
        }
        active_bounds = bounds if bounds else default_bounds
        params = {}
        for key, (min_val, max_val) in active_bounds.items():
            if key.startswith("param_"):
                params[key] = random.uniform(min_val, max_val)
        return params

    def register_new_jobs(self, jobs: List[Dict[str, Any]]):
        """Inserts newly proposed parameters into the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for job in jobs:
                config_hash = job[HASH_KEY]
                gen = job["generation"]
                p1 = job.get("parent_1", None)
                p2 = job.get("parent_2", None)

                cursor.execute(
                    "INSERT OR IGNORE INTO runs (config_hash, generation, status, parent_1, parent_2, origin) VALUES (?, ?, 'pending', ?, ?, ?)",
                    (config_hash, gen, p1, p2, job.get("origin", "NATURAL"))
                )
                cursor.execute(
                    """INSERT OR IGNORE INTO parameters 
                       (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (config_hash, job.get("param_D", 1.0), job.get("param_eta", 0.65), job.get("param_rho_vac", 1.0), job.get("param_a_coupling", 0.5), job.get("param_splash_coupling", 0.5), job.get("param_splash_fraction", -0.5))
                )
            conn.commit()

    def process_generation_results(self, provenance_dir: str, job_hashes: List[str]):
        """Parses provenance JSONs and updates metrics and fitness in the DB with Smooth Gating."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for config_hash in job_hashes:
                prov_path = os.path.join(provenance_dir, f"provenance_{config_hash}.json")
                if not os.path.exists(prov_path):
                    cursor.execute("UPDATE runs SET status='failed' WHERE config_hash=?", (config_hash,))
                    continue
                try:
                    with open(prov_path, 'r') as f:
                        prov_data = json.load(f)
                        
                    spec = prov_data.get("spectral_fidelity", {})
                    aletheia = prov_data.get("aletheia_metrics", {})
                    bridge = prov_data.get("empirical_bridge", {})
                    
                    # Safe Extraction
                    main_sse = float(spec.get("log_prime_sse") or 999.0)
                    null_a = float(spec.get("sse_null_phase_scramble") or 999.0)
                    null_b = float(spec.get("sse_null_target_shuffle") or 999.0)
                    
                    pcs = float(aletheia.get("pcs") or 0.0)
                    pli = float(aletheia.get("pli") or 0.0)
                    ic = float(aletheia.get("ic") or 1.0)
                    
                    j_info = float(aletheia.get("j_info_l2_mean") or 0.0)
                    grad_phase = float(aletheia.get("grad_phase_var_mean") or 0.0)
                    
                    c4_contrast = float(bridge.get("c4_interference_contrast") or 0.0)
                    ablated_c4 = float(bridge.get("ablated_c4_contrast") or 0.0)

                    # --- EPISTEMIC TELEMETRY EXTRACTION ---
                    max_amp = float(aletheia.get("max_amp_peak") or 0.0)
                    clamp_frac = float(aletheia.get("clamp_fraction_mean") or 0.0)
                    omega_sat = float(aletheia.get("omega_sat_mean") or 0.0)
                    
                    # Extract the decoupled metrics from the prov_data payload
                    primary_error = float(spec.get("primary_harmonic_error") or main_sse)
                    miss_penalty = float(spec.get("missing_peak_penalty") or 0.0)
                    noise_pen = float(spec.get("noise_penalty") or 0.0)

                    # --- NEW: Direct Fast Extraction (Zero I/O bottleneck) ---
                    collapse_event_count = int(spec.get("collapse_event_count", 0))

                    # --- COMPOSITE FITNESS ENGINE (For Legacy Metrics/UI) ---
                    if main_sse >= 999.0:
                        fitness = 0.0
                    else:
                        # Fetch the geometric coupling constant directly from DB for the penalty
                        cursor.execute("SELECT param_a_coupling FROM parameters WHERE config_hash=?", (config_hash,))
                        param_row = cursor.fetchone()
                        a_coupling = float(param_row[0]) if param_row else 1.0

                        base_fitness = 1.0 / (main_sse + 1e-9)
                        falsifiability_gap = max(0, null_a - main_sse) + max(0, null_b - main_sse)
                        quantum_falsifiability = max(0, c4_contrast - ablated_c4)
                        
                        # Smoothed, Bounded IC Bonus (Eliminates explosion)
                        ic_bonus = 1.0 / (1.0 + ic)
                        geometry_penalty = grad_phase * 5.0
                        coherence_gate = math.exp(-10.0 * max(0, 0.3 - pcs))
                        
                        # Anti-Flattening Penalty (Forces curved-space interaction)
                        anti_flattening_penalty = math.exp(-20.0 * a_coupling) * 10.0
                        
                        # --- THE EPISTEMIC IMMUNE SYSTEM ---
                        # If the simulation clamps more than 1% of its voxels, or saturates Omega, destroy its fitness.
                        clamp_penalty = clamp_frac * 500.0  
                        omega_sat_penalty = omega_sat * 250.0 
                        
                        raw_fitness = base_fitness + (LAMBDA_CLASSICAL * falsifiability_gap) + (LAMBDA_QUANTUM * quantum_falsifiability) + (pli * 5.0) + ic_bonus - geometry_penalty - anti_flattening_penalty - clamp_penalty - omega_sat_penalty
                        fitness = max(0.0, raw_fitness) * coherence_gate
                    
                    # Update DB (Now includes the LOM telemetry count & decoupled metrics)
                    cursor.execute(
                        """INSERT OR REPLACE INTO metrics 
                           (config_hash, log_prime_sse, primary_harmonic_error, missing_peak_penalty, noise_penalty, 
                            sse_null_phase_scramble, sse_null_target_shuffle, pcs, pli, ic, c4_contrast, ablated_c4_contrast, 
                            j_info_mean, grad_phase_var, max_amp_peak, clamp_fraction_mean, omega_sat_mean, collapse_event_count) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (config_hash, main_sse, primary_error, miss_penalty, noise_pen, 
                         null_a, null_b, pcs, pli, ic, c4_contrast, ablated_c4, 
                         j_info, grad_phase, max_amp, clamp_frac, omega_sat, collapse_event_count)
                    )

                    # --- NEW: PHASE 2 PREDATOR MODE TRIGGER ---
                    if primary_error < 0.01:
                        scaled_peaks = spec.get("scaled_peaks", [])
                        targets = spec.get("prime_log_targets", [])
                        best_target = None
                        
                        if scaled_peaks and targets:
                            # 1. Identify EXACTLY which prime target was hit
                            min_diff = float('inf')
                            for t in targets:
                                for p in scaled_peaks:
                                    if abs(p - t) < min_diff:
                                        min_diff = abs(p - t)
                                        best_target = t
                                        
                        # 2. Fetch the champion's full parameter configuration
                        prev_row_factory = conn.row_factory
                        conn.row_factory = sqlite3.Row
                        cursor.execute("SELECT * FROM parameters WHERE config_hash=?", (config_hash,))
                        param_row = cursor.fetchone()
                        conn.row_factory = prev_row_factory
                        
                        if param_row and best_target is not None:
                            champ_params = dict(param_row)
                            queue_payload = {
                                "champion_hash": config_hash,
                                "champion_params": champ_params,
                                "target_prey_prime": float(best_target),
                                "primary_harmonic_error": primary_error
                            }
                            
                            queue_file = "predator_queue.json"
                            queue_data = []
                            if os.path.exists(queue_file):
                                try:
                                    with open(queue_file, 'r') as qf:
                                        queue_data = json.load(qf)
                                except Exception: pass
                            
                            # Prevent queuing identical twins
                            if not any(q.get("champion_hash") == config_hash for q in queue_data):
                                queue_data.append(queue_payload)
                                
                                # ATOMIC WRITE: Prevents JSON corruption if orchestrator reads mid-write
                                tmp_file = queue_file + ".tmp"
                                with open(tmp_file, 'w') as qf:
                                    json.dump(queue_data, qf, indent=4)
                                os.replace(tmp_file, queue_file)
                                
                                print(f"\n🦅 [HUNTER] PREDATOR LOCK ENGAGED! Target Prime: {best_target:.4f}. Champion sent to queue.")
                    # ------------------------------------------

                    cursor.execute("UPDATE runs SET status='completed', fitness=? WHERE config_hash=?", (fitness, config_hash))
                    
                    # Populate Results Table
                    cursor.execute("""
                        INSERT OR REPLACE INTO results (config_hash, generation, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction, log_prime_sse, fitness, parent_1, parent_2)
                        SELECT r.config_hash, r.generation, p.param_D, p.param_eta, p.param_rho_vac, p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction, m.log_prime_sse, r.fitness, r.parent_1, r.parent_2
                        FROM runs r
                        JOIN parameters p ON r.config_hash = p.config_hash
                        JOIN metrics m ON r.config_hash = m.config_hash
                        WHERE r.config_hash = ?
                    """, (config_hash,))
                    
                except Exception as e:
                    print(f"[Hunter DB] Error parsing {config_hash}: {e}")
                    cursor.execute("UPDATE runs SET status='failed' WHERE config_hash=?", (config_hash,))
            conn.commit()
            
    def get_best_run(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.config_hash, r.fitness, m.log_prime_sse 
                FROM runs r
                JOIN metrics m ON r.config_hash = m.config_hash
                WHERE r.status='completed' 
                ORDER BY r.fitness DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def _get_valid_breeders(self, population: List[Dict[str, Any]]) -> List[int]:
        """Filters the population array to prevent synthetic probes from permanently biasing evolution."""
        valid_indices = []
        for idx, ind in enumerate(population):
            if ind.get("origin", "NATURAL") == "NATURAL":
                valid_indices.append(idx)
            elif ind.get("origin") == "PREDICTOR_ENGINE" and ind.get("rank", 999) == 0:
                valid_indices.append(idx)
        return valid_indices if valid_indices else list(range(len(population)))

    def generate_next_generation(self, population_size: int, bounds: Optional[Dict[str, tuple]] = None) -> List[Dict[str, Any]]:
        current_gen = self.get_current_generation()
        
        with self._get_connection() as conn:
            query = """
                SELECT r.*, p.param_D, p.param_eta, p.param_rho_vac, p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction, 
                       m.log_prime_sse, m.pcs, m.pli, m.ic, m.grad_phase_var,
                       m.primary_harmonic_error, m.missing_peak_penalty, m.noise_penalty, m.collapse_event_count
                FROM runs r
                JOIN parameters p ON r.config_hash = p.config_hash
                LEFT JOIN metrics m ON r.config_hash = m.config_hash
                WHERE r.generation = ? AND r.status = 'completed'
            """
            results = pd.read_sql_query(query, conn, params=(current_gen,))

        if results.empty:
            print(f"[Hunter] No completed runs found in generation {current_gen}. Generating random generation...")
            random_pop = []
            for _ in range(population_size):
                params = self.generate_random_parameters(bounds)
                params['generation'] = current_gen + 1
                params['origin'] = 'NATURAL'
                random_pop.append(params)
            return random_pop

        population = results.to_dict('records')

        # Robust Parsing to prevent NaN breaking sorting
        for ind in population:
            try: ind['log_prime_sse'] = float(ind.get('log_prime_sse', 999.0))
            except: ind['log_prime_sse'] = 999.0
            if math.isnan(ind['log_prime_sse']): ind['log_prime_sse'] = 999.0

            try: ind['pcs'] = float(ind.get('pcs', 0.0))
            except: ind['pcs'] = 0.0
            if math.isnan(ind['pcs']): ind['pcs'] = 0.0

            try: ind['grad_phase_var'] = float(ind.get('grad_phase_var', 999.0))
            except: ind['grad_phase_var'] = 999.0
            if math.isnan(ind['grad_phase_var']): ind['grad_phase_var'] = 999.0

            try: ind['ic'] = float(ind.get('ic', 999.0))
            except: ind['ic'] = 999.0
            if math.isnan(ind['ic']): ind['ic'] = 999.0

            try: ind['missing_peak_penalty'] = float(ind.get('missing_peak_penalty', 999.0))
            except: ind['missing_peak_penalty'] = 999.0
            if math.isnan(ind['missing_peak_penalty']): ind['missing_peak_penalty'] = 999.0

        # --- TRUE MULTI-OBJECTIVE PARETO DOMINANCE (CONSTRAINED NSGA-II) ---
        def dominates(row_a, row_b):
            fit_a = row_a.get('fitness', 0.0)
            fit_b = row_b.get('fitness', 0.0)

            # --- CONSTRAINT GATE (The Epistemic Immune System) ---
            if fit_a > 0.0 and fit_b == 0.0: return True
            if fit_b > 0.0 and fit_a == 0.0: return False

            # --- 5D PARETO GEOMETRY (Independent Dimensions) ---
            err_a = row_a.get('primary_harmonic_error', 999.0)
            err_b = row_b.get('primary_harmonic_error', 999.0)
            
            miss_a = row_a.get('missing_peak_penalty', 999.0)
            miss_b = row_b.get('missing_peak_penalty', 999.0)
            
            pcs_a, pcs_b = row_a.get('pcs', 0.0), row_b.get('pcs', 0.0)
            grad_a, grad_b = row_a.get('grad_phase_var', 999.0), row_b.get('grad_phase_var', 999.0)
            ic_a, ic_b = row_a.get('ic', 999.0), row_b.get('ic', 999.0)

            # A is better or equal on ALL axes
            better_or_eq = (err_a <= err_b) and (miss_a <= miss_b) and (pcs_a >= pcs_b) and (grad_a <= grad_b) and (ic_a <= ic_b)
            # A is strictly better on at least ONE axis
            strictly_better = (err_a < err_b) or (miss_a < miss_b) or (pcs_a > pcs_b) or (grad_a < grad_b) or (ic_a < ic_b)
            
            return better_or_eq and strictly_better

        # --- FAST NON-DOMINATED SORT ---
        fronts: List[List[int]] = [[]]
        S: List[List[int]] = [[] for _ in range(len(population))]
        n = [0] * len(population)

        for p in range(len(population)):
            for q in range(len(population)):
                if dominates(population[p], population[q]):
                    S[p].append(q)
                elif dominates(population[q], population[p]):
                    n[p] += 1
            if n[p] == 0:
                population[p]['rank'] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q]['rank'] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        # --- CROWDING DISTANCE CALCULATOR ---
        # HPC Hardening: Prevent extreme divergence from squashing the diversity scale
        SAFE_MAX = 1000.0 
        SAFE_MIN = -1000.0

        for front in fronts:
            l = len(front)
            if l == 0: continue
            for idx in front:
                population[idx]['crowding_distance'] = 0.0
                
            objectives = ['primary_harmonic_error', 'missing_peak_penalty', 'pcs', 'grad_phase_var', 'ic']
            for obj in objectives:
                sorted_front = sorted(front, key=lambda idx: population[idx][obj])
                
                # Boundaries get infinite distance to ensure survival
                population[sorted_front[0]]['crowding_distance'] = float('inf')
                population[sorted_front[-1]]['crowding_distance'] = float('inf')
                
                # Calculate bounded min and max to protect the denominator
                obj_min = max(population[sorted_front[0]][obj], SAFE_MIN)
                obj_max = min(population[sorted_front[-1]][obj], SAFE_MAX)
                
                val_range = obj_max - obj_min
                if val_range <= 1e-9:
                    continue
                    
                # Assign crowded metrics to interior points
                for j in range(1, l - 1):
                    diff = population[sorted_front[j+1]][obj] - population[sorted_front[j-1]][obj]
                    # Use the clamped val_range to normalize the distance
                    population[sorted_front[j]]['crowding_distance'] += math.fabs(diff) / val_range
                    
        # Persist the True Pareto Frontier (Front 0) to Archive
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx in fronts[0]:
                elite = population[idx]
                cursor.execute("INSERT OR REPLACE INTO pareto_archive (config_hash, generation, log_prime_sse, fitness) VALUES (?, ?, ?, ?)", 
                               (elite['config_hash'], current_gen, elite['log_prime_sse'], elite.get('fitness', 0.0)))
            conn.commit()

        # Sort population by Non-Dominated Rank (Ascending), then by Crowding Distance (Descending)
        pop_indices = list(range(len(population)))
        pop_indices.sort(key=lambda idx: (population[idx]['rank'], -population[idx]['crowding_distance']))

        next_generation = []

        # --- TRUE NSGA-II ELITISM ---
        # The #1 Candidate is now guaranteed to be Rank 0 and occupying the least crowded space on the frontier.
        best_overall = population[pop_indices[0]]
        elite_child = {k: v for k, v in best_overall.items() if k.startswith('param_')}
        elite_child['generation'] = current_gen + 1 # Strict Lineage Fix
        elite_child['parent_1'] = best_overall['config_hash']
        elite_child['parent_2'] = best_overall['config_hash']
        elite_child['origin'] = best_overall.get('origin', 'NATURAL')
        next_generation.append(elite_child)

        # --- NSGA-II CROWDED TOURNAMENT SELECTION ---
        def tournament_select(pop_idx_list, k=TOURNAMENT_SIZE):
            contenders = random.sample(pop_idx_list, min(k, len(pop_idx_list)))
            best = contenders[0]
            for contender in contenders[1:]:
                p_rank = population[contender]['rank']
                b_rank = population[best]['rank']
                
                # 1. Lower Rank Wins
                if p_rank < b_rank:
                    best = contender
                # 2. If Ranks Tie, Larger Crowding Distance Wins (Diversity)
                elif p_rank == b_rank:
                    if population[contender]['crowding_distance'] > population[best]['crowding_distance']:
                        best = contender
            return population[best]

        # --- Adaptive Mutation Decay Schedule ---
        sse_values = results['log_prime_sse'].dropna()
        sse_std = sse_values.std() if len(sse_values) > 1 else 0.0
        decay_factor = max(0.3, 1.0 - current_gen * 0.05)
        active_mutation_strength = MUTATION_STRENGTH * decay_factor
        
        if sse_std < 0.5:
             active_mutation_strength *= 2.0
             print(f"[Hunter] Population converging (SSE std={sse_std:.2f}). Boosting mutation to {active_mutation_strength:.2f}")

        def clamp(val, key):
            default_bounds = {
                "param_D": (0.01, 2.0),
                "param_eta": (0.01, 1.0),
                "param_rho_vac": (0.5, 1.5),
                "param_a_coupling": (0.1, 1.0),
                "param_splash_coupling": (0.0, 1.0),
                "param_splash_fraction": (-1.0, 0.0)
            }
            active_bounds = bounds if bounds else default_bounds
            if key in active_bounds:
                min_val, max_val = active_bounds[key][0], active_bounds[key][1]
                return max(min_val, min(val, max_val))
            return val

        # Pre-Filter Population for Tournament
        valid_breeders_list = self._get_valid_breeders(population)

        # Breeding Loop
        while len(next_generation) < population_size:
            parent1 = tournament_select(valid_breeders_list)
            parent2 = tournament_select(valid_breeders_list)
            attempts = 0
            
            while parent1['config_hash'] == parent2['config_hash'] and attempts < 5 and len(valid_breeders_list) > 1:
                parent2 = tournament_select(valid_breeders_list)
                attempts += 1
                
            p1_params = {k: v for k, v in parent1.items() if k.startswith('param_')}
            p2_params = {k: v for k, v in parent2.items() if k.startswith('param_')}

            child_params = {}
            for key in p1_params.keys():
                # Blended Crossover
                if random.random() < 0.7: 
                    child_params[key] = (p1_params[key] * 0.5) + (p2_params[key] * 0.5)
                else:
                    child_params[key] = p1_params[key]
                
                # Zero-Trap Safe Mutation
                if random.random() < MUTATION_RATE:
                    mutation_factor = random.uniform(-active_mutation_strength, active_mutation_strength)
                    if child_params[key] == 0.0:
                        child_params[key] += mutation_factor * 0.1 
                    else:
                        child_params[key] += child_params[key] * mutation_factor
                        
                child_params[key] = clamp(child_params[key], key)

            # --- PARENTAL LINEAGE LOCK ---
            if parent1.get('origin') == 'PREDICTOR_ENGINE' or parent2.get('origin') == 'PREDICTOR_ENGINE':
                child_params['origin'] = 'PREDICTOR_ENGINE'
                child_params['parent_1'] = 'PREDICTOR_ENGINE'
                child_params['parent_2'] = 'PREDICTOR_ENGINE'
            else:
                child_params['origin'] = 'NATURAL'
                child_params['parent_1'] = parent1['config_hash']
                child_params['parent_2'] = parent2['config_hash']
                
            child_params['generation'] = current_gen + 1
            next_generation.append(child_params)

        print(f"[Hunter] Generated {len(next_generation)} new candidates for generation {current_gen + 1}")
        return next_generation
    
    def add_job(self, job_dict: Dict[str, Any]):
        """Violently explicit database insert."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO runs (config_hash, generation, status, parent_1, parent_2, origin) VALUES (?, ?, 'pending', ?, ?, ?)",
                    (job_dict["config_hash"], job_dict.get("generation", 0), job_dict.get("parent_1"), job_dict.get("parent_2"), job_dict.get("origin", "NATURAL"))
                )
                cursor.execute(
                    """INSERT OR REPLACE INTO parameters 
                    (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (job_dict["config_hash"], job_dict.get("param_D"), job_dict.get("param_eta"), job_dict.get("param_rho_vac"), job_dict.get("param_a_coupling"), job_dict.get("param_splash_coupling", 0.5), job_dict.get("param_splash_fraction", -0.5))
                )
                conn.commit()
        except Exception as e:
            print(f"[Hunter DB ERROR] Failed to insert job! {e}")

    def record_fitness(self, config_hash: str, fitness: float):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE runs SET fitness = ? WHERE config_hash = ?", (fitness, config_hash))
                if cursor.rowcount == 0:
                    print(f"[Hunter DB ERROR] Tried to record fitness for {config_hash[:10]}, but row does not exist in DB!")
                else:
                    print(f"[Hunter DB] Successfully recorded fitness {fitness:.3f} for {config_hash[:10]}")
                conn.commit()
        except Exception as e:
            print(f"[Hunter DB ERROR] Failed to update fitness! {e}")