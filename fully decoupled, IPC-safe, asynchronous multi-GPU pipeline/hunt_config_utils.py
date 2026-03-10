import json
import argparse
from typing import Dict, Any
from pathlib import Path

def load_and_merge_config(config_path: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Loads hunt_config.json and merges with CLI overrides (from argparse.Namespace).
    CLI args take precedence. Returns a single config dict.
    """
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    # Merge CLI overrides (ignore None values)
    cli_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
    config.update(cli_dict)
    return config
