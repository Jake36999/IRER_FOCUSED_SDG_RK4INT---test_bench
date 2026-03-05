from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import os

# --- Pydantic Schemas ---
class ScanRequest(BaseModel):
    version: str
    path: str
    mode: str = "full"
    lmstudio_url: Optional[str] = None
    ai_persona: Optional[str] = None
    rules: Optional[str] = "rules/governance.yaml"
    force_fresh_scan: bool = False
    parameters: Optional[Dict[str, Any]] = None

class ScanResponse(BaseModel):
    version: str
    status: str
    detail: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

router = APIRouter()

SCAN_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "irer-test-bench-scanner", "scanner_main.py"))
SCAN_OUTPUT_PATH = "/tmp/scan_results.json"


@router.post("/scan", response_model=ScanResponse)
async def run_scan(request: ScanRequest):
    # Version check
    if not hasattr(request, 'version') or request.version != '1.0.0':
        raise HTTPException(status_code=400, detail="API contract version mismatch or missing.")
    if not os.path.exists(SCAN_SCRIPT):
        raise HTTPException(status_code=500, detail="Scanner script not found.")

    cmd = [
        "python", SCAN_SCRIPT,
        "--path", request.path,
        "--mode", request.mode,
        "--rules", request.rules,
        "--output-path", SCAN_OUTPUT_PATH
    ]
    if request.lmstudio_url:
        cmd += ["--lmstudio-url", request.lmstudio_url]
    if request.ai_persona:
        cmd += ["--ai-persona", request.ai_persona]
    if request.force_fresh_scan:
        cmd.append("--force-fresh-scan")
    if request.parameters:
        for k, v in request.parameters.items():
            cmd += [f"--{k}", str(v)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Scanner failed: {stderr.decode()}")
        if os.path.exists(SCAN_OUTPUT_PATH):
            with open(SCAN_OUTPUT_PATH, "r", encoding="utf-8") as f:
                import json
                results = json.load(f)
            return ScanResponse(status="success", detail="Scan completed.", results=results)
        else:
            return ScanResponse(status="error", detail="Scan results not found.")
    except Exception as e:
        return ScanResponse(status="error", detail=str(e))
