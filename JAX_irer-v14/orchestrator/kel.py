"""
orchestrator/kel.py
The Knowledge Extraction Log (KEL) Interface.
Handles all interactions with the PostgreSQL Governance Ledger.
"""
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

class KELClient:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "irer_admin")
        self.password = os.getenv("DB_PASS", "irer_secure_password")
        self.dbname = os.getenv("DB_NAME", "irer_kel")
        
        self.conn = None
        self.connect()
        self.initialize_schema()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )
            self.conn.autocommit = True
            print("[KEL] Connected to PostgreSQL Governance Ledger.")
        except Exception as e:
            print(f"[KEL] FATAL: Database connection failed: {e}")
            raise

    def initialize_schema(self):
        """Idempotent schema initialization."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kel_runs (
                    job_id VARCHAR(64) PRIMARY KEY,
                    params JSONB NOT NULL,
                    metrics JSONB,
                    status VARCHAR(20) DEFAULT 'PENDING',
                    artifact_url TEXT,
                    generation INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for fast Genetic Algorithm lookups
                CREATE INDEX IF NOT EXISTS idx_fitness ON kel_runs ((metrics->>'max_h_norm'));
                CREATE INDEX IF NOT EXISTS idx_status ON kel_runs (status);
            """)

    def register_job(self, job_id, params, generation):
        """Logs the Intent-to-Run (Identity-as-Code)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kel_runs (job_id, params, generation, status)
                VALUES (%s, %s, %s, 'PENDING')
                ON CONFLICT (job_id) DO NOTHING;
            """, (job_id, json.dumps(params), generation))

    def update_result(self, job_id, status, metrics, artifact_url):
        """Finalizes the run record."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE kel_runs
                SET status = %s,
                    metrics = %s,
                    artifact_url = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = %s;
            """, (status, json.dumps(metrics), artifact_url, job_id))

    def get_elites(self, limit=10):
        """Fetches top performers (Lowest H-Norm that are STABLE)."""
        # Note: In V14, 'Fitness' is stability (Low H-Norm) + Spectral Fidelity.
        # For now, we optimize for H-Norm stability first.
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT params, metrics 
                FROM kel_runs 
                WHERE status = 'SUCCESS' 
                ORDER BY CAST(metrics->>'max_h_norm' AS FLOAT) ASC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()