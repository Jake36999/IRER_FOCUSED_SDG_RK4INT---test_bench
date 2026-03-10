import hashlib
import json
from typing import Any, Dict

def generate_canonical_hash(config_dict: Dict[str, Any]) -> str:
    """
    Generate a deterministic SHA-256 hash for a configuration dictionary.
    The config is serialized with sorted keys to ensure determinism.
    """
    config_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()
