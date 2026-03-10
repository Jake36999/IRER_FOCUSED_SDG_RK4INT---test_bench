#!/usr/bin/env python3

"""
artifact_bleed_manager.py
Manages artifact bleeding to archive storage with audit logging.
"""

import os
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Set
from concurrent.futures import ThreadPoolExecutor

from orchestrator.diagnostics.runtime_audit import log_lifecycle_event


class ArtifactBleedManager:
    """Manages bleeding of artifacts to archive storage."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.archive_dir = Path(config['archive_dir'])
        self.data_dir = Path(config['data_dir'])
        self.bleed_workers = config.get('bleed_workers', 2)
        self.logger = logging.getLogger(__name__)

        # Thread-safe bleed infrastructure
        self.bleed_lock = threading.Lock()
        self.locked_for_bleed: Set[str] = set()
        self.bleed_executor = ThreadPoolExecutor(max_workers=self.bleed_workers)

        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_part_files()

    def _cleanup_stale_part_files(self) -> None:
        """Remove stale partial archive files left by interrupted bleed operations."""
        stale_seconds = int(self.config.get('stale_part_max_age_seconds', 3600))
        now_ts = time.time()
        removed = 0

        try:
            for part_path in self.archive_dir.rglob("*.part"):
                try:
                    age = now_ts - part_path.stat().st_mtime
                    if age >= stale_seconds:
                        part_path.unlink(missing_ok=True)
                        removed += 1
                except OSError as exc:
                    self.logger.warning(f"Unable to inspect/remove stale partial file {part_path}: {exc}")

            if removed > 0:
                self.logger.info(f"Removed {removed} stale partial archive files from {self.archive_dir}")
        except Exception as exc:
            self.logger.warning(f"Stale partial cleanup skipped: {exc}")

    def queue_for_bleed(self, artifact_path: str, generation: int, config_hash: str):
        """
        Queue an artifact for bleeding to archive.
        """
        try:
            # Emit bleed_queue event
            log_lifecycle_event(
                stage='bleed_queue',
                generation=generation,
                config_hash=config_hash,
                details={"artifact_path": artifact_path}
            )

            # Submit async bleed task
            self.bleed_executor.submit(self._bleed_artifact, artifact_path, generation, config_hash)

            self.logger.info(f"Queued artifact for bleed: {artifact_path}")

        except Exception as e:
            self.logger.error(f"Error queuing artifact for bleed: {e}")

    def _bleed_artifact(self, source_path: str, generation: int, config_hash: str):
        """
        Asynchronously move artifact to archive.
        """
        try:
            source = Path(source_path)
            if not source.exists():
                self.logger.warning(f"Artifact not found for bleed: {source_path}")
                return

            # Determine archive path
            relative_path = source.relative_to(self.data_dir)
            archive_path = self.archive_dir / relative_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            with self.bleed_lock:
                if str(source) in self.locked_for_bleed:
                    self.logger.info(f"Artifact already being bled: {source_path}")
                    return
                self.locked_for_bleed.add(str(source))

            # ASTE D.3: Atomic Finalize to prevent GC race conditions
            part_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
            shutil.move(str(source), str(part_path))
            os.replace(str(part_path), str(archive_path))

            # Emit bleed_archived event
            log_lifecycle_event(
                stage='bleed_archived',
                generation=generation,
                config_hash=config_hash,
                details={"artifact_path": str(archive_path)}
            )

            self.logger.info(f"Successfully bled artifact to: {archive_path}")

        except Exception as e:
            self.logger.error(f"Error bleeding artifact {source_path}: {e}")
        finally:
            with self.bleed_lock:
                self.locked_for_bleed.discard(str(source))

    def get_bleed_status(self) -> Dict[str, Any]:
        """Get current bleed status."""
        return {
            'active_bleeds': len(self.locked_for_bleed),
            'max_workers': self.bleed_workers,
            'archive_dir': str(self.archive_dir)
        }