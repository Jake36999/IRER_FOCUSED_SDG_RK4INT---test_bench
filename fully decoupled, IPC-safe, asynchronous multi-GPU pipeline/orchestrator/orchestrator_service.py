#!/usr/bin/env python3

"""
orchestrator_service.py
Main service entry point for the hardened simulation orchestrator.
Provides CLI interface for diagnostic reporting and orchestrator management.
"""

import argparse
import json
import os
import sys
import time
import signal
from pathlib import Path

# Add orchestrator to path for imports
orchestrator_dir = Path(__file__).parent
sys.path.insert(0, str(orchestrator_dir))

from .diagnostics.lifecycle_report import render_generation_report
from .orchestrator_engine import OrchestratorEngine


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Simulation Orchestrator Service')
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--diagnostic-report',
        action='store_true',
        help='Generate and display diagnostic lifecycle report'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        sys.exit(1)

    if args.diagnostic_report:
        # Generate diagnostic report
        try:
            report = render_generation_report(config)
            print(report)
        except Exception as e:
            print(f"Error generating diagnostic report: {e}")
            sys.exit(1)
    else:
        # Start orchestrator engine
        try:
            engine = OrchestratorEngine(config)
            engine.start()
            print("Orchestrator engine started. Press Ctrl+C to stop.")
            # Keep running until interrupted
            import signal
            def signal_handler(sig, frame):
                print("\nStopping orchestrator engine...")
                engine.stop()
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"Error starting orchestrator engine: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()