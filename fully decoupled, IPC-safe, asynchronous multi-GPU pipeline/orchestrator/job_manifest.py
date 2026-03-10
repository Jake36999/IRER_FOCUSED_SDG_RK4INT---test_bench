"""Job manifest serialization and management."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, Optional
import uuid


class JobManifest:
    """Portable job configuration and identity tracker."""
    
    def __init__(
        self,
        config_hash: str,
        job_id: str,
        generation: int,
        seed: int,
        params: Dict[str, Any],
        origin: str = "NATURAL",
    ):
        self.config_hash = config_hash
        self.job_id = job_id
        self.generation = generation
        self.seed = seed
        self.params = params
        self.origin = origin
    
    @staticmethod
    def from_params(
        params: Dict[str, Any],
        generation: int = 0,
        seed: int = 0,
        origin: str = "NATURAL",
    ) -> JobManifest:
        """Create manifest from parameter dict, computing config_hash."""
        # Compute deterministic config_hash
        param_str = json.dumps(params, sort_keys=True, default=str)
        config_hash = hashlib.sha256(param_str.encode()).hexdigest()
        
        # Generate job_id
        job_id = str(uuid.uuid4())[:16]
        
        return JobManifest(
            config_hash=config_hash,
            job_id=job_id,
            generation=generation,
            seed=seed,
            params=dict(params),
            origin=origin,
        )
    
    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        data = {
            "config_hash": self.config_hash,
            "job_id": self.job_id,
            "generation": self.generation,
            "seed": self.seed,
            "params": self.params,
            "origin": self.origin,
        }
        return json.dumps(data)
    
    @staticmethod
    def from_json(json_str: str) -> JobManifest:
        """Deserialize manifest from JSON string."""
        data = json.loads(json_str)
        return JobManifest(
            config_hash=data["config_hash"],
            job_id=data["job_id"],
            generation=data["generation"],
            seed=data["seed"],
            params=data["params"],
            origin=data.get("origin", "NATURAL"),
        )
