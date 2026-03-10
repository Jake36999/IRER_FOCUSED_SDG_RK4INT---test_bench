"""Artifact garbage collection with generation-aware and archive-gated policies."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Set
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event


def _normalize_path_set(paths: Optional[list]) -> Set[str]:
    """Convert list of paths to absolute path set."""
    if not paths:
        return set()
    return {os.path.abspath(p) for p in paths if p}


def purge_old_artifacts(
    data_dir: str,
    protected_files: Optional[set[str]] = None,
    current_generation: Optional[int] = None,
    locked_files: Optional[set[str]] = None,
    bleed_locked_files: Optional[set[str]] = None,
    archived_files: Optional[set[str]] = None,
    registry: Optional[object] = None,
    min_age_seconds: int = 0,
    require_archived: bool = False,
    target_config_hashes: Optional[Set[str]] = None,
    audit_log_path: Optional[str] = None,
) -> int:
    """Purge old artifacts, optionally requiring archived copy first.
    
    Args:
        data_dir: directory containing .h5 artifacts
        protected_files: set of paths that must not be deleted
        current_generation: if provided, keeps current+previous generation
        locked_files: set of paths currently in use (locked)
        bleed_locked_files: set of paths currently being archived (locked)
        archived_files: set of paths already archived
        registry: artifact registry for generation lookup (optional)
        min_age_seconds: minimum file age before purge eligible
        require_archived: if True, only purge if archived copy exists
        target_config_hashes: if provided, only purge these hashes
        audit_log_path: path to lifecycle audit log
    
    Returns:
        Number of files purged
    """
    archived_set = _normalize_path_set(archived_files or [])
    archived_basenames = {os.path.basename(path) for path in archived_set}
    targets = set(target_config_hashes or [])
    now = time.time()
    purged = 0
    
    if not os.path.exists(data_dir):
        return 0
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".h5"):
            continue
        
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        config_hash = filename.replace("rho_history_", "").replace(".h5", "")
        
        # Skip if target list provided and this hash not in it
        if targets and config_hash not in targets:
            continue
        
        # Never delete already archived
        if filepath in archived_set:
            logging.debug(f"[ArtifactGC] Already archived, skipping: {filename}")
            continue
        
        # If require_archived, skip files without archived twin
        if require_archived and filename not in archived_basenames:
            logging.debug(f"[ArtifactGC] Awaiting archive copy, skipping: {filename}")
            continue
        
        # Skip protected files
        if protected_files and filepath in protected_files:
            logging.debug(f"[ArtifactGC] Protected, skipping: {filename}")
            continue
        
        # Skip locked files
        if (locked_files and filepath in locked_files) or (bleed_locked_files and filepath in bleed_locked_files):
            logging.debug(f"[ArtifactGC] Locked, skipping: {filename}")
            continue
        
        # Skip recent files
        try:
            age = now - os.path.getmtime(filepath)
            if age < min_age_seconds:
                logging.debug(f"[ArtifactGC] Too recent ({age}s < {min_age_seconds}s), skipping: {filename}")
                continue
        except OSError:
            continue
        
        # Generation-aware retention: keep current + previous
        file_generation: Optional[int] = None
        if registry is not None:
            try:
                file_generation = registry.get_generation(config_hash)
            except Exception:
                pass
        
        if file_generation is not None and current_generation is not None:
            age_in_gens = current_generation - file_generation
            if age_in_gens < 2:
                logging.debug(f"[ArtifactGC] Too recent (gen {file_generation}, current {current_generation}), skipping")
                continue
        
        # All checks passed, purge it
        try:
            os.remove(filepath)
            purged += 1
            logging.debug(f"[ArtifactGC] Purged: {filename}")
            log_lifecycle_event(
                stage="gc_purge",
                config_hash=config_hash or None,
                generation=file_generation if current_generation is not None else None,
                details={"filepath": filepath},
                log_path=audit_log_path,
            )
        except OSError as e:
            logging.warning(f"[ArtifactGC] Could not delete {filepath}: {e}")
    
    return purged
