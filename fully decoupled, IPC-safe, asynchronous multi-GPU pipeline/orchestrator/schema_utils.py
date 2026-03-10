"""Schema health check utilities for orchestrator and worker coordination."""
import sqlite3
from typing import Dict, Set, Optional


def get_column_set(db_path: str, table_name: str) -> Set[str]:
    """Get set of column names in a table; return empty set if table doesn't exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        return columns
    except sqlite3.OperationalError:
        return set()


def ensure_ledger_ready(db_path: str, raise_on_fail: bool = False) -> bool:
    """
    Verify ledger has baseline schema (runs, parameters, metrics tables).
    Returns True if schema is ready, False otherwise.
    
    If raise_on_fail is True, raises RuntimeError instead of returning False.
    
    NOTE: This only checks for schema existence. Use initialize_ledger_schema()
    to create schema if missing.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        required = {"runs", "parameters", "metrics"}
        
        if not required.issubset(tables):
            missing = required - tables
            msg = f"Ledger schema incomplete. Missing tables: {missing}"
            conn.close()
            if raise_on_fail:
                raise RuntimeError(msg)
            return False
        
        conn.close()
        return True
    except Exception as e:
        if raise_on_fail:
            raise RuntimeError(f"Ledger health check failed: {e}")
        return False


def initialize_ledger_schema(db_path: str) -> bool:
    """
    Initialize simulation ledger schema if it doesn't exist.
    Creates runs, parameters, and metrics tables with all required columns.
    Returns True if successful, raises RuntimeError on failure.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS runs (
                config_hash TEXT PRIMARY KEY,
                generation INTEGER,
                status TEXT,
                fitness REAL,
                origin TEXT DEFAULT 'NATURAL',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        
        # Create parameters table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS parameters (
                config_hash TEXT PRIMARY KEY,
                param_D REAL,
                param_eta REAL,
                param_rho_vac REAL,
                param_a_coupling REAL,
                param_splash_coupling REAL,
                param_splash_fraction REAL
            )
            '''
        )
        
        # Create metrics table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS metrics (
                config_hash TEXT PRIMARY KEY,
                log_prime_sse REAL,
                bragg_peaks_detected INTEGER,
                n_bragg_peaks INTEGER,
                bragg_prime_sse REAL,
                collapse_event_count INTEGER,
                pcs REAL
            )
            '''
        )
        
        # Add missing columns to runs if needed
        cursor.execute("PRAGMA table_info(runs)")
        runs_columns = {row[1] for row in cursor.fetchall()}
        if 'origin' not in runs_columns:
            cursor.execute("ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'NATURAL'")
        if 'timestamp' not in runs_columns:
            cursor.execute("ALTER TABLE runs ADD COLUMN timestamp DATETIME")
        
        # Add missing columns to metrics if needed
        cursor.execute("PRAGMA table_info(metrics)")
        metrics_columns = {row[1] for row in cursor.fetchall()}
        if 'bragg_peaks_detected' not in metrics_columns:
            cursor.execute("ALTER TABLE metrics ADD COLUMN bragg_peaks_detected INTEGER")
        if 'n_bragg_peaks' not in metrics_columns:
            cursor.execute("ALTER TABLE metrics ADD COLUMN n_bragg_peaks INTEGER")
        if 'collapse_event_count' not in metrics_columns:
            cursor.execute("ALTER TABLE metrics ADD COLUMN collapse_event_count INTEGER")
        if 'pcs' not in metrics_columns:
            cursor.execute("ALTER TABLE metrics ADD COLUMN pcs REAL")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ledger schema: {e}")
