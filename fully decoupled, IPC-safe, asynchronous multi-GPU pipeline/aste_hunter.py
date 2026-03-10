#!/usr/bin/env python3
from typing import List, Dict, Any, Optional, Sequence, Union
import os
import json
import sqlite3
import random
import uuid
import math
import numpy as np
import pandas as pd
from collections import OrderedDict

try:
    from sklearn.cluster import DBSCAN
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    SBD_ENABLED = True
    ASMT_ENABLED = True
except ImportError:
    print("[Hunter V16] Warning: sklearn not found. SBD and ASMT disabled.")
    SBD_ENABLED = False
    ASMT_ENABLED = False

"""
aste_hunter.py (SQLite DB Edition)
CLASSIFICATION: Scientific Adaptive Intelligence Engine (ASTE V16.7 Final Production)
GOAL: Full NSGA-II Implementation + Spectral Gradient Navigation (SGN) + 
      Spectral Basin Detection (SBD) + Adaptive Spectral Manifold Tracking (ASMT).
      Features predictive manifold jumping using Gaussian Process Regression,
      DBSCAN Basin Topological Mapping, Adaptive Mutation Control, 
      Exponential Backoff DB Locking, and Fast Harmonic Prefiltering.
"""

PROVENANCE_DIR = "provenance_reports"

# --- Configuration ---
DB_FILENAME = "simulation_ledger.db"
SSE_METRIC_KEY = "log_prime_sse"
HASH_KEY = "config_hash"
ENABLE_LEGACY_PREDATOR_QUEUE = os.environ.get("ASTE_ENABLE_LEGACY_PREDATOR_QUEUE", "0") == "1"

SQLITE_TIMEOUT = float(os.environ.get("ASTE_SQLITE_TIMEOUT", "30.0"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("ASTE_SQLITE_BUSY_TIMEOUT_MS", "30000"))

# Evolutionary Algorithm Parameters
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05
SGN_PROBABILITY = 0.25  # Base SGN probability (Dynamically altered by SBD)

# Falsifiability Scaling (Decoupled to prevent Quantum/Classical override)
LAMBDA_CLASSICAL = 10.0
LAMBDA_QUANTUM = 5.0


class SpectralManifoldTracker:
    """
    V16 Adaptive Spectral Manifold Tracking (ASMT) Module.
    Uses Gaussian Process Regression to learn the topological mapping
    between spectral harmonic signatures and nonlinear parameter spaces.
    """
    def __init__(self):
        self.trained = False
        if ASMT_ENABLED:
            # Scale-aware kernel scaling for k1, k2, and PCS with explicit bounds to prevent kernel collapse
            kernel = RBF(length_scale=[0.5, 0.5, 0.3], length_scale_bounds=(0.01, 5.0))
            self.model_a = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
            self.model_s = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)

    def train(self, df: pd.DataFrame):
        """Learns the manifold topology from historical spectral metrics."""
        if not ASMT_ENABLED or df is None or len(df) < 20:
            return

        X = df[['dominant_peak_k', 'secondary_peak_k', 'pcs']].values
        
        # Guard against feature collapse (singular kernel matrix)
        if np.std(X[:, 0]) < 1e-3 or np.std(X[:, 1]) < 1e-3:
            print("[ASMT Warning] Feature variance too low. Skipping training.")
            return

        y_a = df['param_a_coupling'].values
        y_s = df['param_splash_coupling'].values

        try:
            self.model_a.fit(X, y_a)
            self.model_s.fit(X, y_s)
            self.trained = True
            print(f"[ASMT Engine] Successfully mapped spectral manifold across {len(X)} states.")
        except Exception as e:
            print(f"[ASMT Warning] GPR Training failed: {e}")

    def predict_parameters(self, k1: float, k2: float, pcs: float) -> Optional[tuple]:
        """Predicts ideal solver parameters for an unexplored spectral target."""
        if not self.trained:
            return None

        X = np.array([[k1, k2, pcs]])
        try:
            # Uncertainty check via predictive variance
            a_mean, a_std = self.model_a.predict(X, return_std=True)
            s_mean, s_std = self.model_s.predict(X, return_std=True)
            
            # Reject if prediction uncertainty is high (std > 0.25)
            if a_std[0] > 0.25 or s_std[0] > 0.25:
                return None
                
            return float(a_mean[0]), float(s_mean[0])
        except:
            return None


class Hunter:
    """
    Implements the core evolutionary 'hunt' logic using a relational SQLite database.
    Manages population parameters, fitness tracking, multi-objective lineage, 
    SGN spatial regressions, Spectral Basin Detection (SBD), and ASMT.
    """

    def export_ledger_to_json(self, output_path: str):
        """[DATA CONTRACT] Dumps the entire SQLite ledger to a formatted JSON file."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                self.execute_with_retry(cursor, "SELECT * FROM results")
                rows = cursor.fetchall()
                ledger_data = [dict(row) for row in rows]
                with open(output_path, 'w') as f:
                    json.dump(ledger_data, f, indent=4)
        except Exception as e:
            print(f"[Hunter Error] Failed to export JSON ledger: {e}")

    def __init__(self, db_file: str = DB_FILENAME):
        self.db_file = db_file
        self._init_db()
        # True LRU implementation via OrderedDict
        self._smoothed_grads = OrderedDict()
        self.gradient_cache = OrderedDict()  # Persistent SGN cross-generation cache
        # Initialize V16 ASMT Tracker
        self.manifold = SpectralManifoldTracker()
        
        self.prefilter_rejections = 0
        self.fast_ratio_rejections = 0
        self.heuristic_rejections = 0
        self.total_processed_runs = 0

    def _get_connection(self):
        """Enforces WAL mode on every single connection for CPU multi-threading safety."""
        conn = sqlite3.connect(self.db_file, timeout=SQLITE_TIMEOUT, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
        return conn

    def execute_with_retry(
        self,
        cursor,
        query: str,
        params: Union[Sequence[Any], Sequence[Sequence[Any]]] = (),
        max_retries: int = 5,
        is_many: bool = False,
    ):
        """Executes a query with exponential backoff to completely eliminate SQLite locked errors."""
        import time
        for attempt in range(max_retries):
            try:
                if is_many:
                    cursor.executemany(query, list(params))
                elif params:
                    cursor.execute(query, tuple(params))
                else:
                    cursor.execute(query)
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                    print(f"[Hunter DB] Database locked. Retrying in {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise

    def _init_db(self):
        """Creates the relational schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Core Run Tracking & Lineage
            self.execute_with_retry(cursor, '''
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
            self.execute_with_retry(cursor, '''
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
            self.execute_with_retry(cursor, '''
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
            self.execute_with_retry(cursor, '''
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
            self.execute_with_retry(cursor, '''
                CREATE TABLE IF NOT EXISTS pareto_archive (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    log_prime_sse REAL,
                    fitness REAL
                )
            ''')

            # --- V14: Spectral Basin Registry ---
            self.execute_with_retry(cursor, '''
                CREATE TABLE IF NOT EXISTS spectral_basins (
                    basin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    k1_mean REAL,
                    k2_mean REAL,
                    pcs_mean REAL,
                    best_error REAL,
                    population_size INTEGER,
                    generation_first_seen INTEGER,
                    generation_last_seen INTEGER,
                    stagnant_generations INTEGER DEFAULT 0
                )
            ''')
            
            # HPC Optimization: Indexing
            self.execute_with_retry(cursor, "CREATE INDEX IF NOT EXISTS idx_generation ON runs(generation);")

            # --- LIVE MIGRATION LOGIC FOR EXISTING DATABASES ---
            try:
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN primary_harmonic_error REAL DEFAULT 999.0")
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN missing_peak_penalty REAL DEFAULT 0.0")
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN noise_penalty REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass 
                
            try:
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN collapse_event_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass 

            try:
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN dominant_peak_k REAL DEFAULT 0.0")
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN secondary_peak_k REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass 

            try:
                self.execute_with_retry(cursor, "ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'NATURAL'")
            except sqlite3.OperationalError:
                pass
                
            try:
                self.execute_with_retry(cursor, "ALTER TABLE metrics ADD COLUMN basin_id INTEGER DEFAULT -1")
            except sqlite3.OperationalError:
                pass
            
            conn.commit()

    def fast_harmonic_prefilter(self, spec: dict, pcs: float) -> bool:
        """
        Cheap harmonic rejection test using spectral peak structure.
        Returns True if the state appears harmonic.
        Returns False if it is likely noise.
        """
        # --- FAST PATH: worker-provided FFT diagnostic ---
        ratio = spec.get("fast_energy_ratio", None)

        if ratio is not None:
            try:
                return float(ratio) >= 4.0
            except (TypeError, ValueError):
                pass

        # --- FALLBACK HEURISTICS ---
        k1 = float(spec.get("dominant_peak_k", 0.0))
        k2 = float(spec.get("secondary_peak_k", 0.0))
        primary_error = float(spec.get("primary_harmonic_error", 999.0))

        # no peak
        if k1 <= 0:
            return False

        # very poor harmonic match
        if primary_error > 5.0:
            return False

        # incoherent spectrum
        if pcs < 0.05:
            return False

        # ensure there is real spectral structure
        peak_separation = abs(k1 - k2)

        if peak_separation < 0.005:
            return False

        return True
            
    def get_current_generation(self) -> int:
        """Finds the highest generation number in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self.execute_with_retry(cursor, "SELECT MAX(generation) FROM runs WHERE status='completed'")
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

                self.execute_with_retry(cursor,
                    "INSERT OR IGNORE INTO runs (config_hash, generation, status, parent_1, parent_2, origin) VALUES (?, ?, 'pending', ?, ?, ?)",
                    (config_hash, gen, p1, p2, job.get("origin", "NATURAL"))
                )
                self.execute_with_retry(cursor,
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
                    self.execute_with_retry(cursor, "UPDATE runs SET status='failed' WHERE config_hash=?", (config_hash,))
                    continue
                try:
                    with open(prov_path, 'r') as f:
                        prov_data = json.load(f)
                        
                    spec = prov_data.get("spectral_fidelity", {})
                    aletheia = prov_data.get("aletheia_metrics", {})
                    
                    # Single extraction point for PCS
                    pcs = float(aletheia.get("pcs") or spec.get("pcs", 0.0))
                    
                    primary_error = float(spec.get("primary_harmonic_error", 999.0))
                    main_sse = float(spec.get("log_prime_sse", primary_error))
                    
                    # --- FIX 2: PREDATOR MODE (Moved above prefilter to prevent skipped attractors) ---
                    if ENABLE_LEGACY_PREDATOR_QUEUE and spec and primary_error < 0.01:
                        scaled_peaks = spec.get("scaled_peaks", [])
                        targets = spec.get("prime_log_targets", [])
                        best_target = None
                        
                        if scaled_peaks and targets:
                            min_diff = float('inf')
                            for t in targets:
                                for p in scaled_peaks:
                                    if abs(p - t) < min_diff:
                                        min_diff = abs(p - t)
                                        best_target = t
                                        
                        prev_row_factory = conn.row_factory
                        conn.row_factory = sqlite3.Row
                        self.execute_with_retry(cursor, "SELECT * FROM parameters WHERE config_hash=?", (config_hash,))
                        param_row_pred = cursor.fetchone()
                        conn.row_factory = prev_row_factory
                        
                        if param_row_pred and best_target is not None:
                            champ_params = dict(param_row_pred)
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
                            
                            if not any(q.get("champion_hash") == config_hash for q in queue_data):
                                queue_data.append(queue_payload)
                                
                                tmp_file = queue_file + ".tmp"
                                with open(tmp_file, 'w') as qf:
                                    json.dump(queue_data, qf, indent=4)
                                os.replace(tmp_file, queue_file)
                                
                                print(f"\n🦅 [HUNTER] PREDATOR LOCK ENGAGED! Target Prime: {best_target:.4f}. Champion sent to queue.")

                    # --- FAST HARMONIC PREFILTER ---
                    self.total_processed_runs += 1
                    
                    if not self.fast_harmonic_prefilter(spec, pcs):
                        self.prefilter_rejections += 1
                        
                        ratio = spec.get("fast_energy_ratio", None)
                        try:
                            if ratio is not None and float(ratio) < 4.0:
                                self.fast_ratio_rejections += 1
                            else:
                                self.heuristic_rejections += 1
                        except (TypeError, ValueError):
                            self.heuristic_rejections += 1
                            
                        # reject early without expensive evaluation
                        fitness = 0.0
                        
                        # Insert dummy metrics row to prevent DB holes for ASMT/SGN historical tracking
                        self.execute_with_retry(cursor,
                            """INSERT OR REPLACE INTO metrics 
                               (config_hash, log_prime_sse, primary_harmonic_error, 
                                dominant_peak_k, secondary_peak_k, pcs) 
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                config_hash,
                                999.0,
                                999.0,
                                float(spec.get("dominant_peak_k", 0.0)),
                                float(spec.get("secondary_peak_k", 0.0)),
                                pcs
                            )
                        )

                        self.execute_with_retry(
                            cursor,
                            "UPDATE runs SET status='completed', fitness=? WHERE config_hash=?",
                            (fitness, config_hash)
                        )

                        continue

                    bridge = prov_data.get("empirical_bridge", {})
                    
                    # Safe Extraction
                    null_a = float(spec.get("sse_null_phase_scramble") or 999.0)
                    null_b = float(spec.get("sse_null_target_shuffle") or 999.0)
                    
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
                    
                    miss_penalty = float(spec.get("missing_peak_penalty") or 0.0)
                    noise_pen = float(spec.get("noise_penalty") or 0.0)

                    collapse_event_count = int(spec.get("collapse_event_count", 0))

                    dominant_peak_k = float(spec.get("dominant_peak_k") or 0.0)
                    secondary_peak_k = float(spec.get("secondary_peak_k") or 0.0)

                    # --- COMPOSITE FITNESS ENGINE ---
                    if main_sse >= 999.0:
                        fitness = 0.0
                    else:
                        self.execute_with_retry(cursor, "SELECT param_a_coupling, param_splash_coupling FROM parameters WHERE config_hash=?", (config_hash,))
                        param_row = cursor.fetchone()
                        a_coupling = float(param_row[0]) if param_row else 1.0

                        base_fitness = 1.0 / (main_sse + 1e-9)
                        falsifiability_gap = max(0, null_a - main_sse) + max(0, null_b - main_sse)
                        quantum_falsifiability = max(0, c4_contrast - ablated_c4)
                        
                        ic_bonus = 1.0 / (1.0 + ic)
                        geometry_penalty = grad_phase * 5.0
                        coherence_gate = math.exp(-10.0 * max(0, 0.3 - pcs))
                        
                        anti_flattening_penalty = math.exp(-20.0 * a_coupling) * 10.0
                        
                        clamp_penalty = clamp_frac * 500.0  
                        omega_sat_penalty = omega_sat * 250.0 
                        
                        raw_fitness = base_fitness + (LAMBDA_CLASSICAL * falsifiability_gap) + (LAMBDA_QUANTUM * quantum_falsifiability) + (pli * 5.0) + ic_bonus - geometry_penalty - anti_flattening_penalty - clamp_penalty - omega_sat_penalty
                        fitness = max(0.0, raw_fitness) * coherence_gate
                    
                    # Update DB 
                    self.execute_with_retry(cursor,
                        """INSERT OR REPLACE INTO metrics 
                           (config_hash, log_prime_sse, primary_harmonic_error, missing_peak_penalty, noise_penalty, 
                            sse_null_phase_scramble, sse_null_target_shuffle, pcs, pli, ic, c4_contrast, ablated_c4_contrast, 
                            j_info_mean, grad_phase_var, max_amp_peak, clamp_fraction_mean, omega_sat_mean, collapse_event_count,
                            dominant_peak_k, secondary_peak_k) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (config_hash, main_sse, primary_error, miss_penalty, noise_pen, 
                         null_a, null_b, pcs, pli, ic, c4_contrast, ablated_c4, 
                         j_info, grad_phase, max_amp, clamp_frac, omega_sat, collapse_event_count,
                         dominant_peak_k, secondary_peak_k)
                    )

                    self.execute_with_retry(cursor, "UPDATE runs SET status='completed', fitness=? WHERE config_hash=?", (fitness, config_hash))
                    
                    self.execute_with_retry(cursor, """
                        INSERT OR REPLACE INTO results (config_hash, generation, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction, log_prime_sse, fitness, parent_1, parent_2)
                        SELECT r.config_hash, r.generation, p.param_D, p.param_eta, p.param_rho_vac, p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction, m.log_prime_sse, r.fitness, r.parent_1, r.parent_2
                        FROM runs r
                        JOIN parameters p ON r.config_hash = p.config_hash
                        JOIN metrics m ON r.config_hash = m.config_hash
                        WHERE r.config_hash = ?
                    """, (config_hash,))
                    
                except Exception as e:
                    print(f"[Hunter DB] Error parsing {config_hash}: {e}")
                    self.execute_with_retry(cursor, "UPDATE runs SET status='failed' WHERE config_hash=?", (config_hash,))
            conn.commit()

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            self.execute_with_retry(cursor, '''
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
        valid_indices = []
        for idx, ind in enumerate(population):
            if ind.get("origin", "NATURAL") == "NATURAL" or ind.get("origin") == "SGN_ENGINE":
                valid_indices.append(idx)
            elif ind.get("origin") in ["PREDICTOR_ENGINE", "FSS_PREDICTOR", "PREDATOR_SWEEP"] and ind.get("rank", 999) == 0:
                valid_indices.append(idx)
        return valid_indices if valid_indices else list(range(len(population)))

    # --- V14 Spectral Basin Detection (SBD) ---
    def detect_spectral_basins(self, population: List[Dict[str, Any]], current_gen: int) -> Dict[int, Dict[str, Any]]:
        """
        DBSCAN phase to organize and detect distinct harmonic families, linking
        them historically within the `spectral_basins` SQLite registry.
        """
        if not SBD_ENABLED:
            return {}

        data = []
        indices = []
        
        # 1. Feature Extraction
        for i, ind in enumerate(population):
            k1 = float(ind.get("dominant_peak_k") or 0.0)
            k2 = float(ind.get("secondary_peak_k") or 0.0)
            pcs = float(ind.get("pcs") or 0.0)
            
            if k1 > 0:
                data.append([k1, k2, pcs])
                indices.append(i)

        if len(data) < 5:
            return {}

        X = np.array(data)
        
        # 2. Local Clustering Phase
        # Enforcing stateless allocation to guarantee zero sklearn attribute retention over long HPC runs
        clustering = DBSCAN(eps=0.08, min_samples=3) if SBD_ENABLED else None
        labels = clustering.fit_predict(X)

        basin_registry = {}
        
        with self._get_connection() as conn:
            existing_basins = pd.read_sql_query("SELECT * FROM spectral_basins", conn)
            cursor = conn.cursor()
            local_to_global = {}

            # 3. Cross-Generation Mapping & Status Updates
            for local_label in set(labels):
                if local_label == -1: 
                    continue

                cluster_indices = [idx for idx, lbl in zip(indices, labels) if lbl == local_label]
                cluster_data = X[[i for i, lbl in enumerate(labels) if lbl == local_label]]

                k1_m = float(np.mean(cluster_data[:, 0]))
                k2_m = float(np.mean(cluster_data[:, 1]))
                pcs_m = float(np.mean(cluster_data[:, 2]))

                cluster_errors = [population[idx].get("primary_harmonic_error", 999.0) for idx in cluster_indices]
                best_err = min(cluster_errors) if cluster_errors else 999.0
                pop_size = len(cluster_indices)

                matched_basin_id = None
                
                if not existing_basins.empty:
                    # Compute Distances on Filtered Subset to prevent DataFrame mismatch bugs
                    valid_matches = existing_basins[
                        (np.abs(existing_basins['k1_mean'] - k1_m) < 0.05) &
                        (np.abs(existing_basins['k2_mean'] - k2_m) < 0.05) &
                        (np.abs(existing_basins['pcs_mean'] - pcs_m) < 0.1)
                    ].copy()
                    
                    if not valid_matches.empty:
                        # Closest historical match
                        valid_matches['total_dist'] = (
                            np.abs(valid_matches['k1_mean'] - k1_m) +
                            np.abs(valid_matches['k2_mean'] - k2_m) +
                            np.abs(valid_matches['pcs_mean'] - pcs_m)
                        )
                        best_match = valid_matches.loc[valid_matches['total_dist'].idxmin()]
                        matched_basin_id = int(best_match['basin_id'])

                if matched_basin_id is not None:
                    # Update Existing Basin
                    row = existing_basins[existing_basins['basin_id'] == matched_basin_id].iloc[0]
                    old_best = float(row['best_error'])
                    stag = int(row['stagnant_generations'])
                    old_pop = int(row['population_size'])

                    if best_err < old_best - 1e-4:
                        new_best = best_err
                        stag = 0
                    else:
                        new_best = old_best
                        stag += 1

                    # Centroid drift tracked securely using population weighting
                    total_pop = old_pop + pop_size
                    new_k1 = (float(row['k1_mean']) * old_pop + k1_m * pop_size) / total_pop
                    new_k2 = (float(row['k2_mean']) * old_pop + k2_m * pop_size) / total_pop
                    new_pcs = (float(row['pcs_mean']) * old_pop + pcs_m * pop_size) / total_pop
                    
                    # Rolling population estimate prevents old_pop from dominating centroid updates
                    new_pop = max(1, int(0.8 * old_pop + 0.2 * pop_size))

                    self.execute_with_retry(cursor, """
                        UPDATE spectral_basins
                        SET k1_mean=?, k2_mean=?, pcs_mean=?, best_error=?, population_size=?, 
                            generation_last_seen=?, stagnant_generations=?
                        WHERE basin_id=?
                    """, (new_k1, new_k2, new_pcs, new_best, new_pop, current_gen, stag, matched_basin_id))
                    
                    local_to_global[local_label] = matched_basin_id
                    basin_registry[matched_basin_id] = {
                        'generation_first_seen': int(row['generation_first_seen']),
                        'generation_last_seen': current_gen,
                        'stagnant_generations': stag,
                        'best_error': new_best,
                        'k1_mean': new_k1,
                        'k2_mean': new_k2,
                        'pcs_mean': new_pcs
                    }
                else:
                    # Discover New Basin
                    self.execute_with_retry(cursor, """
                        INSERT INTO spectral_basins
                        (k1_mean, k2_mean, pcs_mean, best_error, population_size, generation_first_seen, generation_last_seen, stagnant_generations)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (k1_m, k2_m, pcs_m, best_err, pop_size, current_gen, current_gen, 0))
                    
                    new_id = cursor.lastrowid
                    local_to_global[local_label] = new_id
                    basin_registry[new_id] = {
                        'generation_first_seen': current_gen,
                        'generation_last_seen': current_gen,
                        'stagnant_generations': 0,
                        'best_error': best_err,
                        'k1_mean': k1_m,
                        'k2_mean': k2_m,
                        'pcs_mean': pcs_m
                    }

            # 4. Bind resolved basin_id back to current metrics
            metric_updates = []
            
            # Minor Optimization: O(1) lookup map for basin labels
            index_map = {indices[k]: labels[k] for k in range(len(indices))}
            
            for i, ind in enumerate(population):
                glob_id = -1
                if i in index_map:
                    loc_lbl = index_map[i]
                    glob_id = local_to_global.get(loc_lbl, -1)
                
                ind['basin_id'] = glob_id
                metric_updates.append((glob_id, ind['config_hash']))

            self.execute_with_retry(cursor, "UPDATE metrics SET basin_id=? WHERE config_hash=?", metric_updates, is_many=True)
            
            # --- V15.2 Routine Basin Maintenance ---
            # Automatically delete stale, low-population basins that have not been seen in 500+ generations
            # AND prevents deleting stable but temporarily inactive massive attractors
            if current_gen > 0 and current_gen % 500 == 0:
                 self.execute_with_retry(cursor, """
                    DELETE FROM spectral_basins
                    WHERE (? - generation_last_seen) > 500 AND population_size < 3
                 """, (current_gen,))
                 
            conn.commit()

        return basin_registry

    # --- SGN: Spectral Gradient Navigation ---
    def get_sgn_historical_data(self, current_gen: int) -> pd.DataFrame:
        """Fetches the temporal manifold bounding data once per generation."""
        with self._get_connection() as conn:
            target_min_gen = max(0, current_gen - 50)
            # Randomized limit ensures memory and computational stability as history grows
            query = """
                SELECT p.param_a_coupling, p.param_splash_coupling, m.primary_harmonic_error
                FROM parameters p
                JOIN metrics m ON p.config_hash = m.config_hash
                JOIN runs r ON p.config_hash = r.config_hash
                WHERE m.primary_harmonic_error < 1.0
                AND r.generation >= ?
                ORDER BY RANDOM()
                LIMIT 500
            """
            return pd.read_sql_query(query, conn, params=(target_min_gen,))

    def estimate_local_error_gradient(self, df: pd.DataFrame, parent_a: float, parent_s: float) -> Optional[tuple]:
        """
        Estimates local gradient of harmonic error w.r.t parameters using
        Distance-Weighted Ridge-Regularized Local Linear Regression.
        """
        if df is None or len(df) < 12:
            return None
            
        # Linear Regression: e = b0 + b1*a + b2*s
        X = np.column_stack((np.ones(len(df)), df['param_a_coupling'].values, df['param_splash_coupling'].values))
        Y = df['primary_harmonic_error'].values
        
        # Distance-Weighted Regression (sigma = 0.1)
        sigma = 0.1
        dists_sq = (df['param_a_coupling'].values - parent_a)**2 + (df['param_splash_coupling'].values - parent_s)**2
        weights = np.exp(-dists_sq / (sigma**2))
        
        # Weight Normalization to prevent collapse
        weight_sum = np.sum(weights)
        if weight_sum > 1e-12:
            weights /= weight_sum
        else:
            return None
        
        # Vectorized weight application replacing dense np.diag
        Xw = X * weights[:, np.newaxis]
        
        try:
            XtWX = X.T.dot(Xw)
            
            # Regression Condition Gate
            cond_number = np.linalg.cond(XtWX)
            if cond_number > 1e6:
                return None
            
            # Ridge regularization for stabilization
            lambda_ridge = 1e-6
            XtWX += lambda_ridge * np.eye(XtWX.shape[0])
            
            XtWY = Xw.T.dot(Y)
            beta = np.linalg.solve(XtWX, XtWY)
            
            beta0 = float(beta[0])
            grad_a = float(beta[1])
            grad_s = float(beta[2])
            
            # Guard against exploding gradients
            grad_magnitude = math.sqrt(grad_a**2 + grad_s**2)
            if grad_magnitude > 100.0:
                return None
                
            return (beta0, grad_a, grad_s)
            
        except np.linalg.LinAlgError:
            return None
        except Exception as e:
            print(f"[Hunter SGN Warning] Regression failed: {e}")
            return None

    def generate_sgn_candidate(self, base_params: dict, grad_a: float, grad_s: float, learning_rate: float, bounds_dict: dict) -> dict:
        """Applies directed mutation along the harmonic error gradient."""
        child = base_params.copy()
        
        child['param_a_coupling'] -= learning_rate * grad_a
        child['param_splash_coupling'] -= learning_rate * grad_s
        
        # Enforce boundary safety
        for key in ['param_a_coupling', 'param_splash_coupling']:
            if bounds_dict and key in bounds_dict:
                min_v, max_v = bounds_dict[key]
                child[key] = max(min_v, min(child[key], max_v))
                
        return child
    # -----------------------------------------

    def generate_next_generation(self, population_size: int, bounds: Optional[Dict[str, tuple]] = None) -> List[Dict[str, Any]]:
        current_gen = self.get_current_generation()
        
        if self.total_processed_runs > 0:
            rejection_rate = (self.prefilter_rejections / self.total_processed_runs) * 100
            print(
                f"[Hunter] Generation {current_gen}: "
                f"{self.prefilter_rejections}/{self.total_processed_runs} "
                f"rejected ({rejection_rate:.1f}%) "
                f"[FFT Ratio: {self.fast_ratio_rejections}, Heuristic: {self.heuristic_rejections}]"
            )
            self.prefilter_rejections = 0
            self.fast_ratio_rejections = 0
            self.heuristic_rejections = 0
            self.total_processed_runs = 0
            
        default_bounds = {
            "param_D": (0.01, 2.0),
            "param_eta": (0.01, 1.0),
            "param_rho_vac": (0.5, 1.5),
            "param_a_coupling": (0.1, 1.0),
            "param_splash_coupling": (0.0, 1.0),
            "param_splash_fraction": (-1.0, 0.0)
        }
        active_bounds = bounds if bounds else default_bounds
        
        # --- ASMT Manifold Learning Schedule (V15.1 Production Lock) ---
        if current_gen > 0 and current_gen % 10 == 0:
            with self._get_connection() as conn:
                # Tightened coherence filter (PCS > 0.3)
                # Retraining dataset cap (LIMIT 400)
                df_manifold = pd.read_sql_query("""
                    SELECT m.dominant_peak_k, m.secondary_peak_k, m.pcs, 
                           p.param_a_coupling, p.param_splash_coupling
                    FROM metrics m
                    JOIN parameters p ON m.config_hash = p.config_hash
                    WHERE m.dominant_peak_k > 0
                    AND m.pcs > 0.3
                    AND m.primary_harmonic_error < 2.0
                    ORDER BY RANDOM()
                    LIMIT 400
                """, conn)
            self.manifold.train(df_manifold)
            
        if current_gen > 0 and current_gen % 50 == 0:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Optimized Pareto archive pruning using efficient IN clause on DESC sorted set
                self.execute_with_retry(cursor, """
                    DELETE FROM pareto_archive
                    WHERE rowid IN (
                        SELECT rowid
                        FROM pareto_archive
                        ORDER BY fitness DESC
                        LIMIT -1 OFFSET 20000
                    )
                """)
                conn.commit()
            
        with self._get_connection() as conn:
            query = """
                SELECT r.*, p.param_D, p.param_eta, p.param_rho_vac, p.param_a_coupling, p.param_splash_coupling, p.param_splash_fraction, 
                       m.log_prime_sse, m.pcs, m.pli, m.ic, m.grad_phase_var,
                       m.primary_harmonic_error, m.missing_peak_penalty, m.noise_penalty, m.collapse_event_count,
                       m.dominant_peak_k, m.secondary_peak_k, m.basin_id
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
                params = self.generate_random_parameters(active_bounds)
                params['generation'] = current_gen + 1
                params['origin'] = 'NATURAL'
                random_pop.append(params)
            return random_pop

        population = results.to_dict('records')

        for ind in population:
            for k in ['log_prime_sse', 'pcs', 'grad_phase_var', 'ic', 'missing_peak_penalty']:
                try: 
                    ind[k] = float(ind.get(k, 999.0 if k != 'pcs' else 0.0))
                except: 
                    ind[k] = 999.0 if k != 'pcs' else 0.0
                if math.isnan(ind[k]): 
                    ind[k] = 999.0 if k != 'pcs' else 0.0

        # --- Phase 1: V14 Basin Detection and Memory Updates ---
        basin_registry = self.detect_spectral_basins(population, current_gen)

        def dominates(row_a, row_b):
            fit_a = row_a.get('fitness', 0.0)
            fit_b = row_b.get('fitness', 0.0)

            if fit_a > 0.0 and fit_b == 0.0: return True
            if fit_b > 0.0 and fit_a == 0.0: return False

            err_a = row_a.get('primary_harmonic_error', 999.0)
            err_b = row_b.get('primary_harmonic_error', 999.0)
            miss_a = row_a.get('missing_peak_penalty', 999.0)
            miss_b = row_b.get('missing_peak_penalty', 999.0)
            pcs_a, pcs_b = row_a.get('pcs', 0.0), row_b.get('pcs', 0.0)
            grad_a, grad_b = row_a.get('grad_phase_var', 999.0), row_b.get('grad_phase_var', 999.0)
            ic_a, ic_b = row_a.get('ic', 999.0), row_b.get('ic', 999.0)

            better_or_eq = (err_a <= err_b) and (miss_a <= miss_b) and (pcs_a >= pcs_b) and (grad_a <= grad_b) and (ic_a <= ic_b)
            strictly_better = (err_a < err_b) or (miss_a < miss_b) or (pcs_a > pcs_b) or (grad_a < grad_b) or (ic_a < ic_b)
            
            return better_or_eq and strictly_better

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
                
                population[sorted_front[0]]['crowding_distance'] = float('inf')
                population[sorted_front[-1]]['crowding_distance'] = float('inf')
                
                obj_min = max(population[sorted_front[0]][obj], SAFE_MIN)
                obj_max = min(population[sorted_front[-1]][obj], SAFE_MAX)
                
                val_range = obj_max - obj_min
                if val_range <= 1e-9:
                    continue
                    
                for j in range(1, l - 1):
                    diff = population[sorted_front[j+1]][obj] - population[sorted_front[j-1]][obj]
                    population[sorted_front[j]]['crowding_distance'] += math.fabs(diff) / val_range
                    
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx in fronts[0]:
                elite = population[idx]
                self.execute_with_retry(cursor, "INSERT OR REPLACE INTO pareto_archive (config_hash, generation, log_prime_sse, fitness) VALUES (?, ?, ?, ?)", 
                               (elite['config_hash'], current_gen, elite['log_prime_sse'], elite.get('fitness', 0.0)))
            conn.commit()

        pop_indices = list(range(len(population)))
        pop_indices.sort(key=lambda idx: (population[idx]['rank'], -population[idx]['crowding_distance']))

        next_generation = []

        best_overall = population[pop_indices[0]]
        elite_child = {k: v for k, v in best_overall.items() if k.startswith('param_')}
        elite_child['generation'] = current_gen + 1 
        elite_child['parent_1'] = best_overall['config_hash']
        elite_child['parent_2'] = best_overall['config_hash']
        elite_child['origin'] = best_overall.get('origin', 'NATURAL')
        next_generation.append(elite_child)

        def tournament_select(pop_idx_list, k=TOURNAMENT_SIZE):
            contenders = random.sample(pop_idx_list, min(k, len(pop_idx_list)))
            best = contenders[0]
            for contender in contenders[1:]:
                p_rank = population[contender]['rank']
                b_rank = population[best]['rank']
                
                if p_rank < b_rank:
                    best = contender
                elif p_rank == b_rank:
                    if population[contender]['crowding_distance'] > population[best]['crowding_distance']:
                        best = contender
            return population[best]

        sse_values = results['log_prime_sse'].dropna()
        sse_std = sse_values.std() if len(sse_values) > 1 else 0.0
        decay_factor = max(0.3, 1.0 - current_gen * 0.05)
        base_mutation_strength = MUTATION_STRENGTH * decay_factor
        
        if sse_std < 0.5:
             base_mutation_strength *= 2.0

        def clamp(val, key):
            if key in active_bounds:
                min_val, max_val = active_bounds[key][0], active_bounds[key][1]
                return max(min_val, min(val, max_val))
            return val

        valid_breeders_list = self._get_valid_breeders(population)
        
        # --- Pre-calculate Temporal Manifold Data for SGN ---
        historical_df = self.get_sgn_historical_data(current_gen)

        while len(next_generation) < population_size:
            parent1 = tournament_select(valid_breeders_list)
            p1_params = {k: v for k, v in parent1.items() if k.startswith('param_')}
            
            # --- V15 ASMT Predictive Manifold Injection (~7% probability) ---
            if self.manifold.trained and random.random() < 0.07:
                # Sample near known basin centroids instead of uniform random across entire space
                if basin_registry:
                    target_basin = random.choice(list(basin_registry.values()))
                    k1_target = max(0.1, target_basin['k1_mean'] + random.gauss(0, 0.3))
                    k2_target = max(0.1, target_basin['k2_mean'] + random.gauss(0, 0.3))
                    pcs_target = min(1.0, max(0.3, target_basin['pcs_mean'] + random.gauss(0, 0.1)))
                else:
                    k1_target = random.uniform(0.1, 3.0)
                    k2_target = random.uniform(0.1, 3.0)
                    pcs_target = random.uniform(0.3, 1.0)
                
                pred = self.manifold.predict_parameters(k1_target, k2_target, pcs_target)
                if pred:
                    a_pred, s_pred = pred
                    
                    # ASMT Parameter Guard bound-aware check to prevent out-of-manifold jumps
                    a_bounds = active_bounds.get('param_a_coupling', (0.1, 1.0))
                    s_bounds = active_bounds.get('param_splash_coupling', (0.0, 1.0))
                    
                    if not (a_bounds[0] <= a_pred <= a_bounds[1] and s_bounds[0] <= s_pred <= s_bounds[1]):
                        pass # Ignore and fall back to evolution
                    else:
                        child_params = p1_params.copy()  # inherit background physics to ground the leap
                        child_params['param_a_coupling'] = clamp(a_pred, 'param_a_coupling')
                        child_params['param_splash_coupling'] = clamp(s_pred, 'param_splash_coupling')
                        child_params['origin'] = 'ASMT_PREDICTOR'
                        child_params['parent_1'] = 'MANIFOLD'
                        child_params['parent_2'] = 'MANIFOLD'
                        child_params['generation'] = current_gen + 1
                        next_generation.append(child_params)
                        continue

            # --- V14 Adaptive Mutation Controller (SBD Governed) ---
            basin_id = parent1.get('basin_id', -1)
            local_sgn_prob = SGN_PROBABILITY
            local_mut_str = base_mutation_strength
            
            if basin_id != -1 and basin_id in basin_registry:
                b_info = basin_registry[basin_id]
                b_age = b_info['generation_last_seen'] - b_info['generation_first_seen']
                
                if b_info['stagnant_generations'] >= 10:
                    # Escape Mode: Force traversal out of the local well
                    local_sgn_prob = 0.0
                    local_mut_str = base_mutation_strength * 3.0
                elif b_age == 0:
                    # Exploration Mode: Brand new topology detected
                    local_sgn_prob = 0.0
                    local_mut_str = base_mutation_strength * 2.0
                elif b_info['best_error'] < 0.05:
                    # Exploitation Mode: Deep within a highly coherent attractor
                    local_sgn_prob = 0.4
                    local_mut_str = base_mutation_strength * 0.5
            
            # --- SPECTRAL GRADIENT NAVIGATION (SGN) PASS ---
            # SGN Cooldown for ASMT children to prevent destabilization
            if parent1.get('origin') != 'ASMT_PREDICTOR' and \
               historical_df is not None and not historical_df.empty and random.random() < local_sgn_prob:
                
                parent_a = p1_params.get('param_a_coupling', 0.5)
                parent_s = p1_params.get('param_splash_coupling', 0.5)
                
                # Resolving cache key to 4 decimal places for accurate gradient locality
                basin_key = (round(parent_a, 4), round(parent_s, 4))
                
                # Prune memory leak in persistent gradient and smoothing caches (True LRU style)
                if len(self.gradient_cache) > 200:
                    self.gradient_cache.popitem(last=False)
                if len(self._smoothed_grads) > 200:
                    self._smoothed_grads.popitem(last=False)

                # Check cache instead of recomputing
                if basin_key in self.gradient_cache:
                    gradient_data = self.gradient_cache[basin_key]
                    self.gradient_cache.move_to_end(basin_key)
                else:
                    gradient_data = self.estimate_local_error_gradient(historical_df, parent_a, parent_s)
                    self.gradient_cache[basin_key] = gradient_data
                
                if gradient_data:
                    beta0, grad_a, grad_s = gradient_data
                    
                    # Apply Basin-Local Gradient Smoothing (Rolling Average)
                    if basin_key not in self._smoothed_grads:
                        self._smoothed_grads[basin_key] = (grad_a, grad_s)
                        self._smoothed_grads.move_to_end(basin_key)
                    else:
                        prev_grad_a, prev_grad_s = self._smoothed_grads[basin_key]
                        grad_a = 0.7 * prev_grad_a + 0.3 * grad_a
                        grad_s = 0.7 * prev_grad_s + 0.3 * grad_s
                        self._smoothed_grads[basin_key] = (grad_a, grad_s)
                        self._smoothed_grads.move_to_end(basin_key)

                    grad_magnitude = math.sqrt(grad_a**2 + grad_s**2)
                    sgn_learning_rate = 0.01 / (1.0 + grad_magnitude)
                    
                    child_params = self.generate_sgn_candidate(
                        base_params=p1_params, 
                        grad_a=grad_a, 
                        grad_s=grad_s, 
                        learning_rate=sgn_learning_rate,  
                        bounds_dict=active_bounds
                    )
                    
                    # Gradient Direction Verification
                    new_a = child_params['param_a_coupling']
                    new_s = child_params['param_splash_coupling']
                    
                    delta_a = new_a - parent_a
                    delta_s = new_s - parent_s
                    delta_error = grad_a * delta_a + grad_s * delta_s
                    
                    pred_error_new = beta0 + grad_a * new_a + grad_s * new_s
                    actual_parent_error = parent1.get('primary_harmonic_error', 999.0)
                    
                    # Reject if gradient direction increases error OR absolute prediction is worse than parent
                    if delta_error <= 0 and pred_error_new <= actual_parent_error:
                        child_params['origin'] = 'SGN_ENGINE'
                        child_params['parent_1'] = parent1['config_hash']
                        child_params['parent_2'] = parent1['config_hash']
                        child_params['generation'] = current_gen + 1
                        next_generation.append(child_params)
                        continue
                    else:
                        # Fallback to standard mutation implicitly by dropping out of this `if` branch
                        pass
            # -----------------------------------------------

            # --- STANDARD EVOLUTIONARY PASS ---
            parent2 = tournament_select(valid_breeders_list)
            attempts = 0
            while parent1['config_hash'] == parent2['config_hash'] and attempts < 5 and len(valid_breeders_list) > 1:
                parent2 = tournament_select(valid_breeders_list)
                attempts += 1
                
            p2_params = {k: v for k, v in parent2.items() if k.startswith('param_')}

            child_params = {}
            for key in p1_params.keys():
                if random.random() < 0.7: 
                    child_params[key] = (p1_params[key] * 0.5) + (p2_params[key] * 0.5)
                else:
                    child_params[key] = p1_params[key]
                
                # SBD-Governed Local Mutation Applied Here
                if random.random() < MUTATION_RATE:
                    mutation_factor = random.uniform(-local_mut_str, local_mut_str)
                    if child_params[key] == 0.0:
                        child_params[key] += mutation_factor * 0.1 
                    else:
                        child_params[key] += child_params[key] * mutation_factor
                        
                child_params[key] = clamp(child_params[key], key)

            if parent1.get('origin') in ['PREDICTOR_ENGINE', 'ASMT_PREDICTOR', 'FSS_PREDICTOR', 'PREDATOR_SWEEP'] or parent2.get('origin') in ['PREDICTOR_ENGINE', 'ASMT_PREDICTOR', 'FSS_PREDICTOR', 'PREDATOR_SWEEP']:
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
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                self.execute_with_retry(cursor,
                    "INSERT OR REPLACE INTO runs (config_hash, generation, status, parent_1, parent_2, origin) VALUES (?, ?, 'pending', ?, ?, ?)",
                    (job_dict["config_hash"], job_dict.get("generation", 0), job_dict.get("parent_1"), job_dict.get("parent_2"), job_dict.get("origin", "NATURAL"))
                )
                self.execute_with_retry(cursor,
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
                self.execute_with_retry(cursor, "UPDATE runs SET fitness = ? WHERE config_hash = ?", (fitness, config_hash))
                if cursor.rowcount == 0:
                    print(f"[Hunter DB ERROR] Tried to record fitness for {config_hash[:10]}, but row does not exist in DB!")
                else:
                    print(f"[Hunter DB] Successfully recorded fitness {fitness:.3f} for {config_hash[:10]}")
                conn.commit()
        except Exception as e:
            print(f"[Hunter DB ERROR] Failed to update fitness! {e}")