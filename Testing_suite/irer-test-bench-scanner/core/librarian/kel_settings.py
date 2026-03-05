
"""
kel_settings.py
The Centralized Data Contract for the Remedial Manifold.
Bridges the Librarian, Bench, and Triage tools with the God View UI.
"""
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- CATEGORY: CONFIGURATION (The Skeleton) ---
class Configuration:
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	KEL_STORE_ROOT = os.path.join(BASE_DIR, ".kel_store")
	STAGING_ROOT = os.path.join(BASE_DIR, "staging")
	DIRS = {
		"NODES": os.path.join(KEL_STORE_ROOT, "nodes"),
		"CHROMA": os.path.join(KEL_STORE_ROOT, "chroma_db"),
		"INCOMING": os.path.join(STAGING_ROOT, "incoming"),
		"PROCESSED": os.path.join(STAGING_ROOT, "processed"),
		"FAILED": os.path.join(STAGING_ROOT, "failed")
	}
	JAX_VERSION_TARGET = "0.4.20"
	STRICT_MODE = False  # If True, Warn-Only becomes Block-Execution

# --- CATEGORY: TELEMETRY (The Pulse) ---
class Telemetry:
	LOG_LEVEL = "INFO"
	UI_COLOR_MAP = {
		"JAX_PRIMITIVE": "magenta",
		"SCHEMATIC_ERROR": "red",
		"HISTORICAL_INSIGHT": "cyan",
		"REMEDY_FOUND": "green"
	}
	UPDATE_CHANNEL = "kel_telemetry_stream"
	METRICS_KEYS = ["vector_entries", "json_nodes", "unindexed_count"]

# --- CATEGORY: QUERY (The Search) ---
class Query:
	LOOKAHEAD_DEPTH = 10  # Number of cells to scan for a 'fix' after an error
	FAILURE_KEYWORDS = [
		"FAILED", "ERROR", "EXCEPTION", "EXIT CODE", 
		"nan", "inf", "divergence", "collapse", "unstable"
	]
	SEARCH_LIMIT = 3
	SIMILARITY_THRESHOLD = 0.85  # Nodes below this are flagged as UNINDEXED
	RISK_PRIMITIVES = ["jax.lax.scan", "jax.lax.while_loop", "jax.lax.cond"]

# --- USER INPUT DEFINITION (The UI Bridge) ---
class KelUserInput(BaseModel):
	target_path: str = Field(..., description="Path to the file to be scanned or ingested.")
	mode: str = Field("scan", description="One of: 'scan', 'ingest', 'triage'.")
	sensitivity: float = Field(0.85, description="Adjusts the SIMILARITY_THRESHOLD.")
	force_reindex: bool = False
	manual_fix_payload: Optional[Dict[str, Any]] = None

