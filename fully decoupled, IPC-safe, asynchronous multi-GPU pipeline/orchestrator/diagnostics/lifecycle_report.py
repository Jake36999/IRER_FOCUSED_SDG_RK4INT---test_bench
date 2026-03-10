"""Generation-level lifecycle diagnostics report."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

DEFAULT_AUDIT_LOG = os.path.join("runtime_logs", "run_lifecycle_audit.jsonl")


def _load_runs(db_path: str) -> List[Tuple[str, int, str]]:
    """Load all (config_hash, generation, status) from runs table."""
    if not os.path.exists(db_path):
        return []
    rows: List[Tuple[str, int, str]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT config_hash, generation, status FROM runs")
            for cfg_hash, generation, status in cur.fetchall():
                if not cfg_hash:
                    continue
                try:
                    gen = int(generation)
                except (TypeError, ValueError):
                    continue
                rows.append((str(cfg_hash), gen, str(status or "")))
    except Exception:
        return []
    return rows


def _load_purged_counts(audit_path: str) -> Dict[int, int]:
    """Count gc_purge events per generation from audit log."""
    counts: Dict[int, int] = defaultdict(int)
    if not os.path.exists(audit_path):
        return counts
    try:
        with open(audit_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("stage") != "gc_purge":
                    continue
                generation = item.get("generation")
                try:
                    gen = int(generation)
                except (TypeError, ValueError):
                    continue
                counts[gen] += 1
    except Exception:
        return counts
    return counts


def render_generation_report(config: Dict[str, object]) -> str:
    """Render human-readable generation lifecycle table."""
    db_path = str(config.get("db_path", "sqlite_database.db"))
    data_dir = str(config.get("data_dir", "simulation_data"))
    provenance_dir = str(config.get("provenance_dir", "provenance_reports"))
    archive_dir = str(config.get("archive_dir", "archive_runs"))
    audit_path = str(config.get("lifecycle_audit_log", DEFAULT_AUDIT_LOG))

    rows = _load_runs(db_path)
    purged_counts = _load_purged_counts(audit_path)

    by_generation: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {
            "dispatched": 0,
            "terminal": 0,
            "pending": 0,
            "provenance": 0,
            "h5": 0,
            "archived": 0,
            "purged": 0,
        }
    )

    for cfg_hash, generation, status in rows:
        bucket = by_generation[generation]
        bucket["dispatched"] += 1
        status_l = status.lower()
        if status_l == "pending":
            bucket["pending"] += 1
        if status_l in {"completed", "failed", "success", "fail"}:
            bucket["terminal"] += 1

        if os.path.exists(os.path.join(provenance_dir, f"provenance_{cfg_hash}.json")):
            bucket["provenance"] += 1
        if os.path.exists(os.path.join(data_dir, f"rho_history_{cfg_hash}.h5")):
            bucket["h5"] += 1
        if os.path.exists(os.path.join(archive_dir, f"rho_history_{cfg_hash}.h5")):
            bucket["archived"] += 1

    for generation, purged in purged_counts.items():
        by_generation[generation]["purged"] = purged

    lines = []
    lines.append("Generation Lifecycle Report")
    lines.append(f"db_path={db_path}")
    lines.append(f"data_dir={data_dir}")
    lines.append(f"provenance_dir={provenance_dir}")
    lines.append(f"archive_dir={archive_dir}")
    lines.append(f"audit_log={audit_path}")
    lines.append("")
    lines.append(
        "generation | dispatched | terminal | pending | provenance | h5_local | archived | purged"
    )
    lines.append("-" * 92)

    if not by_generation:
        lines.append("no run records found")
        return "\n".join(lines)

    for generation in sorted(by_generation.keys()):
        s = by_generation[generation]
        lines.append(
            f"{generation:10d} | "
            f"{s['dispatched']:10d} | "
            f"{s['terminal']:8d} | "
            f"{s['pending']:7d} | "
            f"{s['provenance']:10d} | "
            f"{s['h5']:8d} | "
            f"{s['archived']:8d} | "
            f"{s['purged']:6d}"
        )

    return "\n".join(lines)
